// BenchMoE.swift — the routed-MoE suites.
//
// The gate/up projection is the one operator the engine runs in two shapes:
// one token at decode, a batch of them at prefill. The bench races the
// paths the engine could actually run, and it reports MLX's gather kernel
// beside them — but never publishes it, because MLX's gather reads a weight
// format the engine does not hold.

import Foundation
import Metal
import QWCore
import QWHAL
import QWQuant

/// The bench's routed-MoE shape. The slab is the model's size, not a
/// token's: the expert count is scaled down instead, because the slab is
/// what the two paths differ on. See `specs/04-qwquant.md`.
enum MoEShape {
    /// How many experts the bank holds.
    static let experts = 4
    /// How many routes a token takes.
    static let routes = 2
    /// How many rows an expert's slab holds — the gate and the up
    /// projection side by side.
    static let rowsPerExpert = 15360
    /// The decode regime: one token.
    static let decodeTokens = 1
    /// The prefill regime: a batch of tokens.
    static let prefillTokens = 16
    /// How many rows the host checks: every stride-th row of every pair.
    static let checkStride = 16
}

/// The weights both formats read. The two formats cannot hold the same
/// bytes, so the bench starts from the function and writes it into both.
enum MoEWeights {
    /// The magic the engine's 4-bit dequant adds to a code before scaling.
    static let magic: Float = 1024.0
    /// The bias the engine's format needs to land on MLX's affine law: the
    /// magic cancels against `magic - 8`, so the stored bias is `-1032 * s`
    /// where MLX's is `-8 * s`.
    static let engineBiasFactor: Float = -1032.0
    /// MLX's affine law's bias, for the same scale.
    static let mlxBiasFactor: Float = -8.0

    /// The expert a (token, route) pair goes to. Skewed on purpose: more
    /// than a third of the pairs land on expert 0, so the grouped paths'
    /// slab reuse pays.
    static func expert(token: Int, route: Int) -> Int {
        if token % 2 == 0 && route == 0 { return 0 }
        return (token + route) % MoEShape.experts
    }

    /// The scale of one 64-value group: a power of two, so it is exact in
    /// f16 and so the products below are exact too.
    static func scale(row: Int, group: Int) -> Float {
        let r = splitmix64(seed(row: row, group: group))
        return 1.0 / Float(1 << (6 + Int(r % 11)))
    }

    /// The 64 codes of one 64-value group. One stream per group, so the
    /// engine's two 32-blocks and MLX's group read the same codes.
    static func codes(row: Int, group: Int) -> [UInt32] {
        var state = seed(row: row, group: group)
        var out = [UInt32]()
        out.reserveCapacity(64)
        for _ in 0..<64 {
            state = splitmix64(state)
            out.append(UInt32(state >> 60))
        }
        return out
    }

    /// The weights in both formats, from one pass over the groups: the
    /// engine's bank and MLX's three planes.
    static func build(rows: Int, cols: Int)
        -> (bank: [UInt8], w: [UInt8], scales: [Float], biases: [Float]) {
        let blocks = cols / 32
        let groups = cols / 64
        let blockBytes = QWAQBlock.bytes(bits: 4, group: 32)
        var bank = [UInt8](repeating: 0, count: rows * blocks * blockBytes)
        var w = [UInt8](repeating: 0, count: rows * groups * 32)
        var scales = [Float](repeating: 0, count: rows * groups)
        var biases = [Float](repeating: 0, count: rows * groups)
        for r in 0..<rows {
            for g in 0..<groups {
                let s = scale(row: r, group: g)
                let values = codes(row: r, group: g)
                scales[r * groups + g] = s
                biases[r * groups + g] = mlxBiasFactor * s
                // MLX's 4-bit layout: the low nibble of byte i is element
                // 2i, the high nibble is element 2i + 1.
                let plane = (r * groups + g) * 32
                for i in 0..<32 {
                    w[plane + i] = UInt8((values[2 * i] & 0xF) | ((values[2 * i + 1] & 0xF) << 4))
                }
                // The engine's bank: two 32-value blocks that share this
                // group's scale.
                for half in 0..<2 {
                    let base = (r * blocks + g * 2 + half) * blockBytes
                    writeF16(&bank, base, s)
                    writeF16(&bank, base + 2, engineBiasFactor * s)
                    for j in 0..<32 {
                        let bit = UInt32(QWAQProducer.hq4NibbleBitOffset(j % 8))
                        let at = base + 4 + (j / 8) * 4 + Int(bit / 8)
                        let shift = Int(bit % 8)
                        let mask = UInt8(0xF << shift)
                        let code = UInt8(values[half * 32 + j] & 0xF)
                        let cleared = bank[at] & ~mask
                        bank[at] = cleared | (code << shift)
                    }
                }
            }
        }
        // The bench writes the bank fast, because the packer allocates a
        // buffer per group and the bank holds millions of them. The writer
        // is checked against the packer, so the fast path cannot drift.
        for r in 0..<min(rows, 2) {
            for b in 0..<min(blocks, 2) {
                let g = b / 2
                let s = scale(row: r, group: g)
                let values = codes(row: r, group: g)
                let half = b % 2
                var packed = [UInt32](repeating: 0, count: 32)
                for j in 0..<32 { packed[j] = values[half * 32 + j] }
                let expected = QWAQProducer.hq4G32Block(codes: packed, scale: s)
                let base = (r * blocks + b) * blockBytes
                for i in 0..<expected.count {
                    precondition(bank[base + i] == expected[i],
                                 "the bench's bank writer disagrees with the incumbent's packer")
                }
            }
        }
        return (bank, w, scales, biases)
    }

    /// The host's answer for one bank row against one activation vector.
    static func referenceRow(bank: [UInt8], bankRow: Int, cols: Int,
                              x: [Float]) -> BenchReference {
        let blocks = cols / 32
        let blockBytes = QWAQBlock.bytes(bits: 4, group: 32)
        var value: Float = 0
        var magnitude: Float = 0
        for b in 0..<blocks {
            let base = (bankRow * blocks + b) * blockBytes
            let s = readF16(bank, base)
            let bias = readF16(bank, base + 2)
            for word in 0..<4 {
                let at = base + 4 + word * 4
                var packed: UInt32 = UInt32(bank[at])
                packed |= UInt32(bank[at + 1]) << 8
                packed |= UInt32(bank[at + 2]) << 16
                packed |= UInt32(bank[at + 3]) << 24
                for j in 0..<8 {
                    let nib = (packed >> UInt32(QWAQProducer.hq4NibbleBitOffset(j))) & 0xF
                    let w = (magic + Float(nib)) * s + bias
                    let term = w * x[b * 32 + word * 8 + j]
                    value += term
                    magnitude += abs(term)
                }
            }
        }
        return BenchReference(value: value, magnitude: magnitude)
    }

    /// Writes an f16 little-endian, the order every host writes it in.
    private static func writeF16(_ bytes: inout [UInt8], _ at: Int, _ value: Float) {
        let bits = Float16(value).bitPattern.littleEndian
        bytes[at] = UInt8(bits & 0xFF)
        bytes[at + 1] = UInt8((bits >> 8) & 0xFF)
    }

    /// Reads an f16 little-endian.
    private static func readF16(_ bytes: [UInt8], _ at: Int) -> Float {
        let bits = UInt16(bytes[at]) | (UInt16(bytes[at + 1]) << 8)
        return Float(Float16(bitPattern: bits))
    }

    /// The seed of one 64-value group: the row and the group, nothing else.
    private static func seed(row: Int, group: Int) -> UInt64 {
        splitmix64(UInt64(row) &* 0x9E37_79B9_7F4A_7C15 &+ UInt64(group) &* 0xBF58_476D_1CE4_E5B9)
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// The routed pairs in expert-major order: MLX's gather reads the weight
/// slab once per run of equal indices, so its rows must be grouped.
struct MoERouting {
    /// The sorted slot each pair sits in.
    let slotOfPair: [Int]
    /// The pair each sorted slot holds.
    let pairOfSlot: [Int]
    /// The expert of each sorted slot.
    let expertOfSlot: [Int]
    /// The token of each sorted slot.
    let tokenOfSlot: [Int]

    /// The order the bench builds from the routing: the pairs grouped by
    /// expert, and stable within an expert.
    static func build(pairs: Int, routes: Int) -> MoERouting {
        var slotOfPair = [Int](repeating: 0, count: pairs)
        var pairOfSlot = [Int](repeating: 0, count: pairs)
        var expertOfSlot = [Int](repeating: 0, count: pairs)
        var tokenOfSlot = [Int](repeating: 0, count: pairs)
        var slot = 0
        for e in 0..<MoEShape.experts {
            for p in 0..<pairs
            where MoEWeights.expert(token: p / routes, route: p % routes) == e {
                slotOfPair[p] = slot
                pairOfSlot[slot] = p
                expertOfSlot[slot] = e
                tokenOfSlot[slot] = p / routes
                slot += 1
            }
        }
        return MoERouting(slotOfPair: slotOfPair, pairOfSlot: pairOfSlot,
                          expertOfSlot: expertOfSlot, tokenOfSlot: tokenOfSlot)
    }
}

/// One checked output row: a row of one routed pair's slab.
struct MoECheck {
    /// The routed pair the row belongs to.
    let pair: Int
    /// The row inside the expert's slab.
    let row: Int
    /// The row's index in a pair-major output.
    let pairMajor: Int
}

/// The two routed-MoE suites.
enum MoESuite: String, CaseIterable, BenchSuite {
    case decode = "moe_gateup_decode"
    case prefill = "moe_gateup_prefill"

    /// How many tokens the suite routes.
    var tokens: Int {
        self == .decode ? MoEShape.decodeTokens : MoEShape.prefillTokens
    }

    /// What a reader has to know about the suite's candidates.
    var note: String {
        switch self {
        case .decode:
            return """
            The decode regime: one token, \(MoEShape.routes) routes, so \
            \(MoEShape.routes) routed pairs. The per-pair GEMV is production's decode path: it \
            reads the expert's slab once per pair. The grouped f32 path is in the repository \
            but nothing dispatches it, and it reads the slab once per expert — which at \
            \(MoEShape.routes) pairs buys nothing and costs a sorting pass. The NAX path is \
            production's *prefill* path: it is reported here but not publishable, because the \
            decode path has f32 activations only and the NAX path reads f16. MLX's gather is \
            bench-only: it reads MLX's three-plane weight format, which the engine never \
            holds. It is timed on rows already grouped by expert, which is what MLX's own \
            pipeline does before it gathers, so its number is the kernel's and not its \
            pipeline's; the sort is timed on its own, so its cost is visible.
            """
        case .prefill:
            return """
            The prefill regime: \(MoEShape.prefillTokens) tokens, \(MoEShape.routes) routes each, so \
            \(MoEShape.prefillTokens * MoEShape.routes) routed pairs, skewed so that more than a third of \
            them land on expert 0. The NAX path is production's prefill path: it reads f16 and \
            writes f16, and its output comes back in expert-major sorted-slot order, so the \
            check reads it through the sort's own map. The two f32 paths read f32 activations \
            and write f32, in pair-major order; the grouped one is in the repository but \
            nothing dispatches it. MLX's gather is bench-only, for the same reason as in the \
            decode suite, and it is timed on rows already grouped by expert.
            """
        }
    }
}

/// The routed-MoE suites.
enum MoESuites {
    /// The verification gate, the same one the other suites use.
    static let gate = 1e-3

    /// Runs one routed-MoE suite and returns what it measured.
    static func run(hal: QWHal, suite: MoESuite, cols: Int,
                     samples: Int, repeats: Int) throws -> OperatorSuites.OperatorResult {
        let device = hal.device
        let hidden = cols
        let tokens = suite.tokens
        let routes = MoEShape.routes
        let experts = MoEShape.experts
        let rowsPerExpert = MoEShape.rowsPerExpert
        let pairs = tokens * routes
        // MLX's gather takes its two alignment flags as function constants,
        // so they are fixed for every shape the bench dispatches. This is
        // the check that the shape is the one they describe.
        precondition(rowsPerExpert % 64 == 0 && hidden % 64 == 0,
                     "MLX's gather assumes 64-wide extents in N and K")

        // --- the weights, in both formats ---
        let weights = MoEWeights.build(rows: experts * rowsPerExpert, cols: hidden)
        let bankBuf = try BenchRunner.buffer(device, bytes: weights.bank.count)
        weights.bank.withUnsafeBytes { src in
            bankBuf.contents().copyMemory(from: src.baseAddress!, byteCount: weights.bank.count)
        }
        let mlxW = try BenchRunner.buffer(device, bytes: weights.w.count)
        weights.w.withUnsafeBytes { src in
            mlxW.contents().copyMemory(from: src.baseAddress!, byteCount: weights.w.count)
        }
        let mlxScales = try BenchRunner.buffer(device, bytes: weights.scales.count * 2)
        BenchRunner.fillF16(mlxScales, weights.scales)
        let mlxBiases = try BenchRunner.buffer(device, bytes: weights.biases.count * 2)
        BenchRunner.fillF16(mlxBiases, weights.biases)

        // --- the activations ---
        // One vector per token: the engine's rotated hidden state is a
        // matrix, and the routed paths read a token's row.
        let xs = BenchRunner.makeActivations(cols: tokens * hidden)
        let x32 = try BenchRunner.buffer(device, bytes: tokens * hidden * 4)
        BenchRunner.fillF32(x32, xs)
        let x16 = try BenchRunner.buffer(device, bytes: tokens * hidden * 2)
        BenchRunner.fillF16(x16, xs)
        // MLX's gather reads one activation row per routed pair, in the
        // expert-major order the bench builds below.
        let routing = MoERouting.build(pairs: pairs, routes: routes)
        var sortedXs = [Float](repeating: 0, count: pairs * hidden)
        for slot in 0..<pairs {
            let t = routing.tokenOfSlot[slot]
            for j in 0..<hidden { sortedXs[slot * hidden + j] = xs[t * hidden + j] }
        }
        let xSorted16 = try BenchRunner.buffer(device, bytes: pairs * hidden * 2)
        BenchRunner.fillF16(xSorted16, sortedXs)

        // --- the routing ---
        let ids = try BenchRunner.buffer(device, bytes: pairs * 4)
        let idView = ids.contents().bindMemory(to: Int32.self, capacity: pairs)
        let mlxIndices = try BenchRunner.buffer(device, bytes: pairs * 4)
        let indexView = mlxIndices.contents().bindMemory(to: UInt32.self, capacity: pairs)
        for p in 0..<pairs {
            let e = UInt32(MoEWeights.expert(token: p / routes, route: p % routes))
            idView[p] = Int32(e)
        }
        for slot in 0..<pairs { indexView[slot] = UInt32(routing.expertOfSlot[slot]) }

        // --- the counting sort's outputs ---
        let expertOff = try BenchRunner.buffer(device, bytes: (experts + 1) * 4)
        let sortedPairIdx = try BenchRunner.buffer(device, bytes: pairs * 4)
        let sortedTok = try BenchRunner.buffer(device, bytes: pairs * 4)
        let invPerm = try BenchRunner.buffer(device, bytes: pairs * 4)
        let maxTiles = max(1, min(pairs, experts + (pairs + 15) / 16))
        let tileE = try BenchRunner.buffer(device, bytes: maxTiles * 4)
        let tileR = try BenchRunner.buffer(device, bytes: maxTiles * 4)
        let tileCount = try BenchRunner.buffer(device, bytes: 4)

        // --- the outputs ---
        let outF32 = try BenchRunner.buffer(device, bytes: pairs * rowsPerExpert * 4)
        let outGroup = try BenchRunner.buffer(device, bytes: pairs * rowsPerExpert * 4)
        let outNAX = try BenchRunner.buffer(device, bytes: pairs * rowsPerExpert * 2)
        let outMLX = try BenchRunner.buffer(device, bytes: pairs * rowsPerExpert * 2)

        // --- the host's answer ---
        // Every stride-th row of every pair: the slab holds a million rows,
        // and the host's answer for all of them costs more than the bench.
        var checks: [MoECheck] = []
        checks.reserveCapacity(pairs * (rowsPerExpert / MoEShape.checkStride))
        for p in 0..<pairs {
            var row = 0
            while row < rowsPerExpert {
                checks.append(MoECheck(pair: p, row: row, pairMajor: p * rowsPerExpert + row))
                row += MoEShape.checkStride
            }
        }
        var expected = [BenchReference](repeating: BenchReference(value: 0, magnitude: 0),
                                        count: checks.count)
        for (i, c) in checks.enumerated() {
            let t = c.pair / routes
            let e = MoEWeights.expert(token: t, route: c.pair % routes)
            let x = Array(xs[t * hidden..<(t + 1) * hidden])
            expected[i] = MoEWeights.referenceRow(bank: weights.bank,
                                                  bankRow: e * rowsPerExpert + c.row,
                                                  cols: hidden, x: x)
        }

        // --- the pipelines ---
        let psoPerPair = try hal.pipeline("qwg_moe_gateup_q4g_f32_r1")
        let psoSort = try hal.pipeline("qwg_moe_group_pairs_f32")
        let psoGroup = try hal.pipeline("qwg_moe_gateup_group_f32")
        let psoNAX = try hal.pipeline("qwg_moe_nax_gemm")
        // MLX picks the row block from the number of rows per expert; the
        // bench takes the same rule, so the candidate is upstream's kernel
        // at upstream's tile, not a strawman.
        let mlxRowBlock = pairs / experts < 64 ? 32 : 64
        let mlxKernelName = mlxRowBlock == 32
            ? "qwg_moe_gather_mlx_nax_bm32" : "qwg_moe_gather_mlx_nax_bm64"
        let psoMLX = try hal.pipeline(mlxKernelName)

        // --- the candidates ---
        // The counting sort: which pair went to which expert, and the tile
        // list the NAX GEMM walks.
        let encodeSort: (any MTLComputeCommandEncoder) -> Void = { enc in
            enc.setComputePipelineState(psoSort)
            enc.setBuffer(ids, offset: 0, index: 0)
            enc.setBuffer(expertOff, offset: 0, index: 1)
            enc.setBuffer(sortedPairIdx, offset: 0, index: 2)
            enc.setBuffer(sortedTok, offset: 0, index: 3)
            var nExperts = UInt32(experts), totalPairs = UInt32(pairs), topK = UInt32(routes)
            enc.setBytes(&nExperts, length: 4, index: 4)
            enc.setBytes(&totalPairs, length: 4, index: 5)
            enc.setBytes(&topK, length: 4, index: 6)
            enc.setBuffer(invPerm, offset: 0, index: 7)
            var writeInv: UInt32 = 1
            enc.setBytes(&writeInv, length: 4, index: 8)
            enc.setBuffer(tileE, offset: 0, index: 9)
            enc.setBuffer(tileR, offset: 0, index: 10)
            enc.setBuffer(tileCount, offset: 0, index: 11)
            var writeTiles: UInt32 = 1
            enc.setBytes(&writeTiles, length: 4, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // The engine's per-pair GEMV: one dispatch, f32 in and out, the
        // output in pair-major order.
        let encodePerPair: (any MTLComputeCommandEncoder) -> Void = { enc in
            enc.setComputePipelineState(psoPerPair)
            enc.setBuffer(bankBuf, offset: 0, index: 0)
            enc.setBuffer(x32, offset: 0, index: 1)
            enc.setBuffer(ids, offset: 0, index: 2)
            enc.setBuffer(outF32, offset: 0, index: 3)
            var n = UInt32(rowsPerExpert), k = UInt32(hidden), e = UInt32(routes)
            enc.setBytes(&n, length: 4, index: 4)
            enc.setBytes(&k, length: 4, index: 5)
            enc.setBytes(&e, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: (rowsPerExpert + 3) / 4, height: routes, depth: tokens),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        // The engine's grouped GEMM: the sort, then one grouped GEMM per
        // expert, f32 in and out, the output in pair-major order.
        let encodeGrouped: (any MTLComputeCommandEncoder) -> Void = { enc in
            encodeSort(enc)
            enc.setComputePipelineState(psoGroup)
            enc.setBuffer(bankBuf, offset: 0, index: 0)
            enc.setBuffer(x32, offset: 0, index: 1)
            enc.setBuffer(outGroup, offset: 0, index: 2)
            enc.setBuffer(expertOff, offset: 0, index: 3)
            enc.setBuffer(sortedPairIdx, offset: 0, index: 4)
            enc.setBuffer(sortedTok, offset: 0, index: 5)
            var n = UInt32(rowsPerExpert), k = UInt32(hidden), r = UInt32(routes)
            enc.setBytes(&n, length: 4, index: 6)
            enc.setBytes(&k, length: 4, index: 7)
            enc.setBytes(&r, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: experts, height: (rowsPerExpert + 31) / 32, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }

        // The engine's NAX GEMM: the sort, then the tiled GEMM. It reads f16
        // activations and writes f16, in expert-major sorted-slot order.
        let encodeNAX: (any MTLComputeCommandEncoder) -> Void = { enc in
            encodeSort(enc)
            enc.setComputePipelineState(psoNAX)
            enc.setBuffer(bankBuf, offset: 0, index: 0)
            enc.setBuffer(x16, offset: 0, index: 1)
            enc.setBuffer(outNAX, offset: 0, index: 2)
            enc.setBuffer(expertOff, offset: 0, index: 3)
            enc.setBuffer(tileE, offset: 0, index: 4)
            enc.setBuffer(tileR, offset: 0, index: 5)
            enc.setBuffer(tileCount, offset: 0, index: 6)
            var params = MoEGroupedParams(experts: UInt32(experts), n: UInt32(rowsPerExpert),
                                          k: UInt32(hidden), blocksPerRow: UInt32(hidden / 32),
                                          rowTiles: 0, colTiles: UInt32(rowsPerExpert / 64))
            enc.setBytes(&params, length: MemoryLayout<MoEGroupedParams>.size, index: 7)
            var tileBlock: UInt32 = 0
            enc.setBytes(&tileBlock, length: 4, index: 8)
            enc.setBuffer(sortedTok, offset: 0, index: 9)
            var hasGather: UInt32 = 1
            enc.setBytes(&hasGather, length: 4, index: 10)
            enc.dispatchThreadgroups(
                MTLSize(width: rowsPerExpert / 64, height: maxTiles, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }

        // MLX's gather: one dispatch, f16 in and out, the output in pair-major
        // order. It reads MLX's own weight format, so the engine cannot run
        // it as it stands.
        let encodeMLX: (any MTLComputeCommandEncoder) -> Void = { enc in
            enc.setComputePipelineState(psoMLX)
            enc.setBuffer(xSorted16, offset: 0, index: 0)
            enc.setBuffer(mlxW, offset: 0, index: 1)
            enc.setBuffer(mlxScales, offset: 0, index: 2)
            enc.setBuffer(mlxBiases, offset: 0, index: 3)
            enc.setBuffer(mlxIndices, offset: 0, index: 4)
            enc.setBuffer(outMLX, offset: 0, index: 5)
            var m = Int32(pairs), n = Int32(rowsPerExpert), k = Int32(hidden)
            enc.setBytes(&m, length: 4, index: 6)
            enc.setBytes(&n, length: 4, index: 7)
            enc.setBytes(&k, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: (rowsPerExpert + 63) / 64,
                        height: (pairs + mlxRowBlock - 1) / mlxRowBlock, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        }

        // The engine's decode path has f32 activations only; the prefill path
        // has both. So the NAX path is a candidate the engine can only run at
        // prefill.
        let naxIsRunnable = suite == .prefill
        let racers = [
            OperatorSuites.Racer(name: "qwg_moe_gateup_q4g_f32_r1", dispatches: 1,
                                 encode: { enc in encodePerPair(enc) }),
            OperatorSuites.Racer(name: "qwg_moe_group_pairs_f32 + qwg_moe_gateup_group_f32",
                                 dispatches: 2, publishable: true,
                                 encode: { enc in encodeGrouped(enc) }),
            OperatorSuites.Racer(name: "qwg_moe_group_pairs_f32 + qwg_moe_nax_gemm",
                                 dispatches: 2, publishable: naxIsRunnable,
                                 encode: { enc in encodeNAX(enc) }),
            OperatorSuites.Racer(name: mlxKernelName, dispatches: 1,
                                 publishable: false, encode: { enc in encodeMLX(enc) }),
            OperatorSuites.Racer(name: "qwg_moe_group_pairs_f32", dispatches: 1,
                                 publishable: false, encode: { enc in encodeSort(enc) }),
        ]

        return try OperatorSuites.race(hal: hal, suite: suite, rows: rowsPerExpert, cols: hidden,
                                       racers: racers,
                                       check: { which in
                                           switch which {
                                           case 0:
                                               return worstF32(outF32, checks, expected,
                                                                count: pairs * rowsPerExpert) { $0.pairMajor }
                                           case 1:
                                               return worstF32(outGroup, checks, expected,
                                                                count: pairs * rowsPerExpert) { $0.pairMajor }
                                           case 2:
                                               // The NAX output is in expert-major
                                               // sorted-slot order, so it is read
                                               // through the sort's own map.
                                               let slot = invPerm.contents()
                                                   .bindMemory(to: Int32.self, capacity: pairs)
                                               return worstF16(outNAX, checks, expected,
                                                                count: pairs * rowsPerExpert) { c in
                                                   Int(slot[c.pair]) * rowsPerExpert + c.row
                                               }
                                           case 3:
                                               // MLX's output is in the expert-major
                                               // order the bench fed it, so it is read
                                               // through that order.
                                               return worstF16(outMLX, checks, expected,
                                                                count: pairs * rowsPerExpert) { c in
                                                   routing.slotOfPair[c.pair] * rowsPerExpert + c.row
                                               }
                                           default:
                                               // The sort alone: every pair appears
                                               // once, and its token is the pair's.
                                               let sorted = sortedPairIdx.contents()
                                                   .bindMemory(to: Int32.self, capacity: pairs)
                                               let toks = sortedTok.contents()
                                                   .bindMemory(to: Int32.self, capacity: pairs)
                                               var seen = Set<Int>()
                                               for at in 0..<pairs {
                                                   let p = Int(sorted[at])
                                                   guard seen.insert(p).inserted,
                                                         Int(toks[at]) == p / routes else { return 1 }
                                               }
                                               return 0
                                           }
                                       },
                                       samples: samples, repeats: repeats)
    }

    /// The worst relative error of an f32 output against the reference.
    private static func worstF32(_ buffer: any MTLBuffer, _ checks: [MoECheck],
                                  _ reference: [BenchReference], count: Int,
                                  map: (MoECheck) -> Int) -> Double {
        let got = buffer.contents().bindMemory(to: Float.self, capacity: count)
        var worst: Double = 0
        for (i, c) in checks.enumerated() {
            worst = max(worst, BenchRunner.relativeError(got: got[map(c)], reference: reference[i]))
        }
        return worst
    }

    /// The worst relative error of an f16 output against the reference.
    private static func worstF16(_ buffer: any MTLBuffer, _ checks: [MoECheck],
                                  _ reference: [BenchReference], count: Int,
                                  map: (MoECheck) -> Int) -> Double {
        let got = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        var worst: Double = 0
        for (i, c) in checks.enumerated() {
            let v = Float(Float16(bitPattern: got[map(c)]))
            worst = max(worst, BenchRunner.relativeError(got: v, reference: reference[i]))
        }
        return worst
    }
}

/// The grouped GEMM's parameters, as `QWGemmGroupedParams` lays them out.
struct MoEGroupedParams {
    let experts: UInt32
    let n: UInt32
    let k: UInt32
    let blocksPerRow: UInt32
    let rowTiles: UInt32
    let colTiles: UInt32
}
