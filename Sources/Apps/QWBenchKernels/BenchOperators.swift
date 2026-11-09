// BenchOperators.swift — the per-operator suites.
//
// Each suite races two ways of doing one operator, on the operator's own
// shape. A candidate is VERIFIED against the host before it is timed, at the
// shape it is timed at: a number for a kernel that computes the wrong thing
// is worse than no number at all. Every binding is the production one.

import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

/// What the bench needs to know about the suite it is reporting. The dense
/// operators and the routed-MoE ones answer the same questions, so they share
/// the harness.
protocol BenchSuite {
    /// The operation the suite's entries are filed under.
    var rawValue: String { get }
    /// What a reader has to know about this suite's candidates.
    var note: String { get }
}

/// The operators the bench races.
enum OperatorSuite: String, CaseIterable, BenchSuite {
    case attnQKV = "attn_qkv"
    case mlpSwiglu = "mlp_swiglu"
    case gdnInproj = "gdn_inproj_decode"
    case lmHead = "lm_head"

    /// What a reader has to know about this suite's candidates. The report
    /// prints it with the suite's section: a candidate that does more work
    /// than its rival is not faster, it is different.
    var note: String {
        switch self {
        case .attnQKV:
            return "The fragmented side is three dispatches of the same kernel over the "
                + "q/k/v sub-views — a composition the repository does not run; the q/k/v "
                + "projection is concatenated into one bank at load time."
        case .mlpSwiglu:
            return "The fragmented side is the decode GEMV over the 2I bank with the "
                + "prefill epilogue — a hybrid that exists nowhere in the repository. "
                + "Production decode is the fused pair; production prefill is a tiled "
                + "GEMM with the same epilogue."
        case .gdnInproj:
            return "The fused candidate also writes 96 rows of `glog`/`betas` control "
                + "output that the plain GEMVs do not, so the two sides are not "
                + "work-equivalent."
        case .lmHead:
            return "Each candidate folds a different number of rows into one key pair, "
                + "so its key buffers hold a different number of slots."
        }
    }
}

/// The operator suites, and the shape each one races on.
enum OperatorSuites {
    /// The suites, in the order the bench reports them.
    static let all: [OperatorSuite] = [.attnQKV, .mlpSwiglu, .gdnInproj, .lmHead]

    /// Every operator suite races the incumbent's production kernels, so it
    /// reads the incumbent's 4-bit G32 container — the same 20 bytes per 32
    /// columns, the magic's fold in the header.
    static let bits = 4
    static let group = 32

    /// The verification gate, the same one the family sweep uses.
    static let gate = 1e-3

    /// The QKV projection's width, and the three sub-views it is made of:
    /// 6144 q + 6144 gate + 1024 k + 1024 v.
    static let qkvRows = 14336
    static let qkvSplit = [12288, 1024, 1024]

    /// The MLP's intermediate width. The bank holds gate and up, so it is
    /// twice this wide.
    static let mlpIntermediate = 8192

    /// The GDN input projection's width. The kernel's own layout pins it:
    /// 16384 qkvz rows, then 48 `a` rows and 48 `b` rows.
    static let gdnRows = 16480
    static let gdnQKVZ = 16384

    /// The LM head's width — a bench-scale vocabulary, not the model's.
    static let headRows = 32768

    /// Runs one suite and returns its measured points.
    static func run(hal: QWHal, suite: OperatorSuite, cols: Int,
                    samples: Int, repeats: Int) throws -> OperatorResult {
        switch suite {
        case .attnQKV:
            return try attnQKV(hal: hal, cols: cols, samples: samples, repeats: repeats)
        case .mlpSwiglu:
            return try mlpSwiglu(hal: hal, cols: cols, samples: samples, repeats: repeats)
        case .gdnInproj:
            return try gdnInproj(hal: hal, cols: cols, samples: samples, repeats: repeats)
        case .lmHead:
            return try lmHead(hal: hal, cols: cols, samples: samples, repeats: repeats)
        }
    }

    /// One candidate in a race: the kernels it dispatches.
    struct Racer {
        let name: String
        /// How many dispatches the candidate makes. The report states it.
        let dispatches: Int
        /// The candidate's key-slot count, when it has one.
        let slots: Int?
        /// Whether the engine could run this candidate as it stands.
        let publishable: Bool
        let encode: (any MTLComputeCommandEncoder) -> Void

        init(name: String, dispatches: Int, slots: Int? = nil,
             publishable: Bool = true,
             encode: @escaping (any MTLComputeCommandEncoder) -> Void) {
            self.name = name
            self.dispatches = dispatches
            self.slots = slots
            self.publishable = publishable
            self.encode = encode
        }
    }

    /// One candidate's measurement, as the report states it.
    struct OperatorCandidate {
        let name: String
        /// How many dispatches it makes.
        let dispatches: Int
        /// The key-slot count, when it has one.
        let slots: Int?
        /// Whether the engine could run this candidate as it stands.
        let publishable: Bool
        /// The worst it disagreed with the host by.
        let error: Double
        /// The median of its timed samples.
        let median: Double
        /// Its fastest sample.
        let low: Double
        /// Its slowest sample.
        let high: Double
    }

    /// Checks every candidate, then times them alternately, so a clock ramp
    /// lands on all of them. Every candidate gets its own control, and the
    /// gap between a kernel and its own re-measurement is the noise floor.
    static func race(hal: QWHal, suite: any BenchSuite, rows: Int, cols: Int,
                     racers: [Racer], check: (Int) throws -> Double,
                     samples: Int, repeats: Int) throws -> OperatorResult {
        var errors = [Double](repeating: 0, count: racers.count)
        for (i, racer) in racers.enumerated() {
            _ = try BenchRunner.time(hal: hal, name: racer.name,
                                      warm: { enc in racer.encode(enc) },
                                      timed: { enc in racer.encode(enc) })
            errors[i] = try check(i)
            guard errors[i] <= gate else {
                throw BenchError.verificationFailed(racer.name, errors[i])
            }
        }
        // Every candidate is measured a second time, as a racer of its own.
        // One control would not do: the floor has to cover the candidate that
        // wins, and not only the one that came first in the list.
        let controls = racers.map { racer in
            Racer(name: racer.name + " (control)", dispatches: racer.dispatches,
                  slots: racer.slots, publishable: racer.publishable,
                  encode: racer.encode)
        }
        let all = racers + controls
        var times = [[Double]](repeating: [], count: all.count)
        for round in 0..<samples {
            // The order rotates, so no racer is always first after the flush
            // and none is systematically favoured by its position.
            for step in 0..<all.count {
                let i = (round + step) % all.count
                times[i].append(try BenchRunner.time(hal: hal, name: all[i].name,
                                                      warm: { enc in all[i].encode(enc) },
                                                      timed: { enc in all[i].encode(enc) },
                                                      repeats: repeats))
            }
        }
        let medians = times.map { BenchRunner.median($0) }
        // The candidates the engine could run are the ones the decision is
        // about, so they are the ones the floor and the separation rank.
        let publishable = racers.indices.filter { racers[$0].publishable }
        // The noise floor: the largest gap between a candidate and its own
        // re-measurement. The largest, not the average — a floor has to
        // cover the worst the measurement can produce.
        var noise: Double = 0
        for i in publishable {
            let control = medians[racers.count + i]
            if control > 0 {
                noise = max(noise, abs(medians[i] - control) / control)
            }
        }
        // The separation: the gap between the fastest candidate and the
        // runner-up. Only the candidates the engine could run are ranked.
        var separation: Double = 0
        var separable = false
        if publishable.count >= 2 {
            let ranked = publishable.sorted { (medians[$0], $0) < (medians[$1], $1) }
            let fastest = medians[ranked[0]]
            let runnerUp = medians[ranked[1]]
            separation = runnerUp == 0 ? 0 : abs(runnerUp - fastest) / runnerUp
            // An entry is published only when the separation is larger than
            // the noise floor. Inside it, the two are equivalent, and the
            // engine should keep the path it already runs — so the map says
            // nothing.
            separable = separation > noise
        }
        let candidates = racers.indices.map { i in
            OperatorCandidate(name: racers[i].name, dispatches: racers[i].dispatches,
                               slots: racers[i].slots, publishable: racers[i].publishable,
                               error: errors[i], median: medians[i],
                               low: times[i].min() ?? medians[i],
                               high: times[i].max() ?? medians[i])
        }
        let points = separable ? publishable.map { i in
            BenchPoint(nBucket: rows, op: suite.rawValue, bits: bits, group: group,
                       candidate: racers[i].name, gpuMilliseconds: medians[i])
        } : []
        return OperatorResult(op: suite.rawValue, rows: rows, cols: cols,
                              note: suite.note,
                              candidates: candidates, points: points, noise: noise,
                              separation: separation, separable: separable)
    }

    /// What one suite measured: its candidates, its noise floor and whether
    /// the candidates could be separated at all.
    struct OperatorResult {
        /// The operation the suite measured.
        let op: String
        /// The output-row count it measured at.
        let rows: Int
        /// The reduction width it measured at.
        let cols: Int
        /// What a reader has to know about the suite's candidates.
        let note: String
        let candidates: [OperatorCandidate]
        /// The points the winner map takes — empty when the suite could not
        /// separate the candidates.
        let points: [BenchPoint]
        /// The gap between a candidate and itself, as a fraction.
        let noise: Double
        /// The gap between the fastest candidate and the runner-up.
        let separation: Double
        /// Whether the separation is larger than the noise floor.
        let separable: Bool
    }

    /// The bank's row stride, in bytes.
    static func rowBytes(cols: Int) -> Int {
        (cols / group) * QWAQBlock.bytes(bits: bits, group: group)
    }

    /// The host's answer for one row of the bank, and the magnitude the
    /// error against it is measured on.
    static func reference(bank: Data, row: Int, cols: Int, xs: [Float]) -> BenchReference {
        AQBanks.incumbentReferenceRow(row: row, cols: cols, bank: bank, x: xs)
    }

    // MARK: - The attention QKV projection

    /// One wide GEMV against three narrow ones, over the same bank. The
    /// fused bank is what production loads; the fragmented side is three
    /// dispatches of the same kernel over the three sub-views.
    private static func attnQKV(hal: QWHal, cols: Int,
                                 samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let rows = qkvRows
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let y = try BenchRunner.buffer(device, bytes: rows * 2)
        var expected = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                        count: rows)
        for r in 0..<rows {
            expected[r] = reference(bank: bank, row: r, cols: cols, xs: xs)
        }
        let pso = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
        let stride = rowBytes(cols: cols)

        /// One dispatch over `count` rows starting at `row0`.
        func encode(_ enc: any MTLComputeCommandEncoder, row0: Int, count: Int) {
            enc.setComputePipelineState(pso)
            enc.setBuffer(weights, offset: row0 * stride, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            // The kernel declares an f32 output it never writes; production
            // binds the f16 tensor there, and so does the bench.
            enc.setBuffer(y, offset: row0 * 2, index: 3)
            var n = UInt32(count), k = UInt32(cols)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&k, length: 4, index: 5)
            enc.setBuffer(y, offset: row0 * 2, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (count + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        var views: [(row0: Int, count: Int)] = []
        var at = 0
        for count in qkvSplit { views.append((at, count)); at += count }

        let racers = [
            Racer(name: "qwg_gemv_q4g_dx_h16_t4", dispatches: 1,
                  encode: { enc in encode(enc, row0: 0, count: rows) }),
            Racer(name: "qwg_gemv_q4g_dx_h16_t4 x3", dispatches: 3,
                  encode: { enc in for v in views { encode(enc, row0: v.row0, count: v.count) } }),
        ]
        return try race(hal: hal, suite: OperatorSuite.attnQKV, rows: rows, cols: cols, racers: racers,
                        check: { _ in BenchRunner.worstError(y, rows: rows, reference: expected) },
                        samples: samples, repeats: repeats)
    }

    // MARK: - The dense MLP's SwiGLU

    /// The fused `silu_pair` against the fragmented GEMV plus `silu_mul`.
    /// The fused path needs a separate rotation dispatch; the fragmented
    /// path fuses the rotation into `silu_mul`. Both include the rotation.
    private static func mlpSwiglu(hal: QWHal, cols: Int,
                                  samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let i = mlpIntermediate
        let rows = 2 * i
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let h16 = try BenchRunner.buffer(device, bytes: i * 2)
        let gu = try BenchRunner.buffer(device, bytes: rows * 2)
        let out = try BenchRunner.buffer(device, bytes: i * 2)

        // The host's answer: the SwiGLU product, then the rotation.
        var products = [Float](repeating: 0, count: i)
        for r in 0..<i {
            let gate = reference(bank: bank, row: r, cols: cols, xs: xs).value
            let up = reference(bank: bank, row: r + i, cols: cols, xs: xs).value
            products[r] = gate / (1.0 + exp(-gate)) * up
        }
        var rotated = products
        BenchRunner.fwht32(&rotated)
        var expected = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0), count: i)
        for r in 0..<i {
            // The magnitude is the group's L2 norm: the rotation is
            // orthonormal, so it is the scale the output lives on.
            let group = (r / 32) * 32
            var norm: Float = 0
            for j in group..<(group + 32) { norm += products[j] * products[j] }
            expected[r] = BenchReference(value: rotated[r], magnitude: norm.squareRoot())
        }

        let psoPair = try hal.pipeline("qwg_gemv_q4g_silu_pair_dx_h16")
        let psoFwht = try hal.pipeline("qwg_fwht32_f16")
        let psoSilu = try hal.pipeline("qwg_silu_mul_f16")
        let psoGemv = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")

        let fused = Racer(name: "qwg_gemv_q4g_silu_pair_dx_h16 + qwg_fwht32_f16",
                          dispatches: 2,
                          encode: { enc in
                              enc.setComputePipelineState(psoPair)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x, offset: 0, index: 1)
                              enc.setBuffer(h16, offset: 0, index: 3)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 4)
                              enc.setBytes(&k, length: 4, index: 5)
                              enc.setBuffer(h16, offset: 0, index: 8)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: (i + 3) / 4, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                              enc.setComputePipelineState(psoFwht)
                              enc.setBuffer(h16, offset: 0, index: 0)
                              var count = UInt32(i)
                              enc.setBytes(&count, length: 4, index: 1)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: i / 32, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                          })

        let fragmented = Racer(name: "qwg_gemv_q4g_dx_h16_t4 + qwg_silu_mul_f16",
                               dispatches: 2,
                               encode: { enc in
                                   enc.setComputePipelineState(psoGemv)
                                   enc.setBuffer(weights, offset: 0, index: 0)
                                   enc.setBuffer(x, offset: 0, index: 1)
                                   enc.setBuffer(gu, offset: 0, index: 3)
                                   var n = UInt32(rows), k = UInt32(cols)
                                   enc.setBytes(&n, length: 4, index: 4)
                                   enc.setBytes(&k, length: 4, index: 5)
                                   enc.setBuffer(gu, offset: 0, index: 8)
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                                   enc.setComputePipelineState(psoSilu)
                                   enc.setBuffer(gu, offset: 0, index: 0)
                                   enc.setBuffer(out, offset: 0, index: 1)
                                   var m = UInt32(1), iu = UInt32(i)
                                   enc.setBytes(&m, length: 4, index: 2)
                                   enc.setBytes(&iu, length: 4, index: 3)
                                   enc.dispatchThreads(
                                       MTLSize(width: (i + 7) / 8, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                               })

        return try race(hal: hal, suite: OperatorSuite.mlpSwiglu, rows: rows, cols: cols, racers: [fused, fragmented],
                        check: { which in
                            which == 0
                                ? BenchRunner.worstError(h16, rows: i, reference: expected)
                                : BenchRunner.worstError(out, rows: i, reference: expected)
                        },
                        samples: samples, repeats: repeats)
    }

    // MARK: - The GDN input projection

    /// The fused GDN projection against the two plain GEMVs it replaces. The
    /// fused kernel also writes the 96 rows of control output — `glog` and
    /// `betas` — that the plain GEMVs do not.
    private static func gdnInproj(hal: QWHal, cols: Int,
                                   samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let rows = gdnRows
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let y = try BenchRunner.buffer(device, bytes: rows * 2)
        let y32 = try BenchRunner.buffer(device, bytes: rows * 4)

        // The control inputs: 48 `a` and 48 `b` rows, each with its own
        // log-scale and bias.
        let aLog = try BenchRunner.buffer(device, bytes: 48 * 4)
        let dtBias = try BenchRunner.buffer(device, bytes: 48 * 4)
        let betas = try BenchRunner.buffer(device, bytes: 48 * 4)
        let glog = try BenchRunner.buffer(device, bytes: 48 * 4)
        var aLogValues = [Float](repeating: 0, count: 48)
        var dtBiasValues = [Float](repeating: 0, count: 48)
        for h in 0..<48 {
            aLogValues[h] = Float(h % 7) * 0.125 - 0.375
            dtBiasValues[h] = Float(h % 5) * 0.25 - 0.5
        }
        BenchRunner.fillF32(aLog, aLogValues)
        BenchRunner.fillF32(dtBias, dtBiasValues)

        var expected = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                        count: rows)
        for r in 0..<rows {
            expected[r] = reference(bank: bank, row: r, cols: cols, xs: xs)
        }

        let psoGdn = try hal.pipeline("qwg_gemv_q4g_dx_h16_gdn_inproj_t4")
        let psoGemv = try hal.pipeline("qwg_gemv_q4g_dx_h16_t4")
        let stride = rowBytes(cols: cols)

        func encodeGemv(_ enc: any MTLComputeCommandEncoder, row0: Int, count: Int) {
            enc.setComputePipelineState(psoGemv)
            enc.setBuffer(weights, offset: row0 * stride, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            // Production binds the **f16** tensor to this dead slot; the
            // kernel never writes it, and neither does the bench.
            enc.setBuffer(y, offset: 0, index: 3)
            var n = UInt32(count), k = UInt32(cols)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&k, length: 4, index: 5)
            enc.setBuffer(y, offset: row0 * 2, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: (count + 3) / 4, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        let fused = Racer(name: "qwg_gemv_q4g_dx_h16_gdn_inproj_t4",
                          dispatches: 1,
                          encode: { enc in
                              enc.setComputePipelineState(psoGdn)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x, offset: 0, index: 1)
                              enc.setBuffer(y32, offset: 0, index: 3)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 4)
                              enc.setBytes(&k, length: 4, index: 5)
                              enc.setBuffer(y, offset: 0, index: 8)
                              enc.setBuffer(aLog, offset: 0, index: 9)
                              enc.setBuffer(dtBias, offset: 0, index: 10)
                              enc.setBuffer(betas, offset: 0, index: 11)
                              enc.setBuffer(glog, offset: 0, index: 12)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                          })

        let fragmented = Racer(name: "qwg_gemv_q4g_dx_h16_t4 x2",
                               dispatches: 2,
                               encode: { enc in
                                   encodeGemv(enc, row0: 0, count: gdnQKVZ)
                                   encodeGemv(enc, row0: gdnQKVZ, count: rows - gdnQKVZ)
                               })

        return try race(hal: hal, suite: OperatorSuite.gdnInproj, rows: rows, cols: cols, racers: [fused, fragmented],
                        check: { which in
                            var worst = BenchRunner.worstError(y, rows: rows, reference: expected)
                            if which == 0 {
                                // The epilogue, checked against the formula
                                // applied to the kernel's **own** vx: the dot
                                // product is verified on its own, above.
                                let device = y32.contents().bindMemory(to: Float.self, capacity: rows)
                                let gotGlog = glog.contents().bindMemory(to: Float.self, capacity: 48)
                                let gotBetas = betas.contents().bindMemory(to: Float.self, capacity: 48)
                                for h in 0..<48 {
                                    let va = device[gdnQKVZ + h]
                                    let vb = device[gdnQKVZ + 48 + h]
                                    let wantGlog = -exp(aLogValues[h])
                                        * log(1.0 + exp(va + dtBiasValues[h]))
                                    let wantBetas = 1.0 / (1.0 + exp(-vb))
                                    worst = max(worst, rel(gotGlog[h], wantGlog))
                                    worst = max(worst, rel(gotBetas[h], wantBetas))
                                }
                            }
                            return worst
                        },
                        samples: samples, repeats: repeats)
    }

    /// The relative error of two values, against the larger of them and a floor.
    private static func rel(_ got: Float, _ want: Float) -> Double {
        let scale = max(abs(Double(want)), 1e-6)
        return abs(Double(got) - Double(want)) / scale
    }

    // MARK: - The LM head and the argmax

    /// The dense engine's fused head against its fragmented one. The check is
    /// the **token**, not the key slots: the slot layout is an implementation
    /// detail of the fused path.
    private static func lmHead(hal: QWHal, cols: Int,
                               samples: Int, repeats: Int) throws -> OperatorResult {
        let device = hal.device
        let rows = headRows
        let bank = AQBanks.makeIncumbent4G32(rows: rows, cols: cols)
        let weights = try BenchRunner.buffer(device, bytes: bank.count)
        bank.withUnsafeBytes { src in
            weights.contents().copyMemory(from: src.baseAddress!, byteCount: bank.count)
        }
        let xs = BenchRunner.makeActivations(cols: cols)
        let x = try BenchRunner.buffer(device, bytes: cols * 2)
        BenchRunner.fillF16(x, xs)
        let x32 = try BenchRunner.buffer(device, bytes: cols * 4)
        BenchRunner.fillF32(x32, xs)
        let logits = try BenchRunner.buffer(device, bytes: rows * 4)
        let keysHi = try BenchRunner.buffer(device, bytes: rows * 4)
        let keysLo = try BenchRunner.buffer(device, bytes: rows * 4)
        let token = try BenchRunner.buffer(device, bytes: 4)

        // The host's answer: the largest logit, ties to the lowest row.
        var bestRow = 0
        var bestValue = -Float.infinity
        var values = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            let v = reference(bank: bank, row: r, cols: cols, xs: xs).value
            values[r] = v
            if v > bestValue { bestValue = v; bestRow = r }
        }

        let psoHead = try hal.pipeline("qwg_head_gemv_q4g_argmax")
        let psoArgmax = try hal.pipeline("qwg_head_argmax")
        let psoFin = try hal.pipeline("qwg_head_argmax_fin")
        let psoGemv = try hal.pipeline("qwg_gemv_q4g_v4")
        let psoFlash = try hal.pipeline("qwg_gemv_q4g_f32_v4_am")

        // The fused path: one key pair per threadgroup of eight rows.
        let headSlots = (rows + 7) / 8
        let fused = Racer(name: "qwg_head_gemv_q4g_argmax + qwg_head_argmax_fin",
                          dispatches: 2, slots: headSlots,
                          encode: { enc in
                              enc.setComputePipelineState(psoHead)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x, offset: 0, index: 1)
                              enc.setBuffer(logits, offset: 0, index: 3)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 4)
                              enc.setBytes(&k, length: 4, index: 5)
                              enc.setBuffer(keysHi, offset: 0, index: 6)
                              enc.setBuffer(keysLo, offset: 0, index: 7)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: headSlots, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                              enc.memoryBarrier(scope: .buffers)
                              enc.setComputePipelineState(psoFin)
                              enc.setBuffer(keysHi, offset: 0, index: 0)
                              enc.setBuffer(keysLo, offset: 0, index: 1)
                              enc.setBuffer(token, offset: 0, index: 2)
                              var slots = UInt32(headSlots)
                              enc.setBytes(&slots, length: 4, index: 3)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                          })

        // The fragmented path: the logits, then a separate argmax.
        let argmaxSlots = (rows + 255) / 256
        let fragmented = Racer(name: "qwg_gemv_q4g_v4 + qwg_head_argmax + qwg_head_argmax_fin",
                               dispatches: 3, slots: argmaxSlots,
                               encode: { enc in
                                   enc.setComputePipelineState(psoGemv)
                                   enc.setBuffer(weights, offset: 0, index: 0)
                                   enc.setBuffer(x, offset: 0, index: 1)
                                   enc.setBuffer(logits, offset: 0, index: 3)
                                   var n = UInt32(rows), k = UInt32(cols)
                                   enc.setBytes(&n, length: 4, index: 4)
                                   enc.setBytes(&k, length: 4, index: 5)
                                   // Production's `handQ4gV4` geometry: four
                                   // rows per threadgroup, 128 threads.
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                                   enc.setComputePipelineState(psoArgmax)
                                   enc.setBuffer(logits, offset: 0, index: 0)
                                   enc.setBuffer(keysHi, offset: 0, index: 1)
                                   enc.setBuffer(keysLo, offset: 0, index: 2)
                                   var n2 = UInt32(rows)
                                   enc.setBytes(&n2, length: 4, index: 3)
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: argmaxSlots, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                   enc.memoryBarrier(scope: .buffers)
                                   enc.setComputePipelineState(psoFin)
                                   enc.setBuffer(keysHi, offset: 0, index: 0)
                                   enc.setBuffer(keysLo, offset: 0, index: 1)
                                   enc.setBuffer(token, offset: 0, index: 2)
                                   var slots = UInt32(argmaxSlots)
                                   enc.setBytes(&slots, length: 4, index: 3)
                                   enc.dispatchThreadgroups(
                                       MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                               })

        // The FlashNext path: f32 activations, sixteen rows folded into one
        // key pair per threadgroup, and no separate argmax pass.
        let flashSlots = (rows + 15) / 16
        let flash = Racer(name: "qwg_gemv_q4g_f32_v4_am + qwg_head_argmax_fin",
                          dispatches: 2, slots: flashSlots,
                          encode: { enc in
                              enc.setComputePipelineState(psoFlash)
                              enc.setBuffer(weights, offset: 0, index: 0)
                              enc.setBuffer(x32, offset: 0, index: 1)
                              enc.setBuffer(logits, offset: 0, index: 2)
                              var n = UInt32(rows), k = UInt32(cols)
                              enc.setBytes(&n, length: 4, index: 3)
                              enc.setBytes(&k, length: 4, index: 4)
                              var epi: UInt32 = 0
                              enc.setBytes(&epi, length: 4, index: 5)
                              var scale: Float = 1
                              enc.setBytes(&scale, length: 4, index: 6)
                              enc.setBuffer(keysHi, offset: 0, index: 7)
                              enc.setBuffer(keysLo, offset: 0, index: 8)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: flashSlots, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                              enc.memoryBarrier(scope: .buffers)
                              enc.setComputePipelineState(psoFin)
                              enc.setBuffer(keysHi, offset: 0, index: 0)
                              enc.setBuffer(keysLo, offset: 0, index: 1)
                              enc.setBuffer(token, offset: 0, index: 2)
                              var slots = UInt32(flashSlots)
                              enc.setBytes(&slots, length: 4, index: 3)
                              enc.dispatchThreadgroups(
                                  MTLSize(width: 1, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                          })
        return try race(hal: hal, suite: OperatorSuite.lmHead, rows: rows, cols: cols,
                        racers: [fused, fragmented, flash],
                        check: { _ in
                            let got = token.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
                            guard Int(got) < rows else { return Double.infinity }
                            if Int(got) == bestRow { return 0 }
                            // A different token is only a failure when the two
                            // logit values are not the same to rounding.
                            let scale = max(abs(Double(bestValue)), 1e-6)
                            let gap = abs(Double(bestValue) - Double(values[Int(got)])) / scale
                            return gap
                        },
                        samples: samples, repeats: repeats)
    }
}
