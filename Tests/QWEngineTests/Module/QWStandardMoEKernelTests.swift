// QWStandardMoEKernelTests.swift — the standard MoE path's compute.
//
// The routed layer's chain is checked against a CPU reference that runs the
// same maths, in the STANDARD basis: the rotation is an involution, so the
// packed weight times the rotated activation IS the standard product.
// The test also breaks the chain on purpose, one stage at a time.

import XCTest
import Foundation
import Metal
@testable import QWEngine
@testable import QWCore
@testable import QWQuant
@testable import QWHAL
@testable import QWAlloc
@testable import QWGEMM

final class QWStandardMoEKernelTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    // MARK: - The synthetic model

    /// A sparse MoE geometry: the fixture's, with the model's own flag.
    static func geometry(normTopKProb: Bool) throws -> QWModelGeometry {
        try QWModelGeometry.stubPureMoE(normTopKProb: normTopKProb)
    }

    /// A synthetic hq4 bank: one scale, a bias from a small set, so two
    /// experts' rows differ by far more than the format's 1.5% nibble step.
    static func makeBank(rows: Int, k: Int, seed: inout UInt64) -> Data {
        let biases: [UInt16] = [0x0000, 0x2800, 0x2C00, 0x3000, 0x3400, 0x3800, 0xB800, 0xBC00]
        var d = Data(capacity: rows * (k / 32) * 20)
        func next() -> UInt64 {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return seed
        }
        for _ in 0..<(rows * (k / 32)) {
            let bias = biases[Int((next() >> 33) % UInt64(biases.count))]
            d.append(Data([UInt8(0x00), UInt8(0x04)]))          // scale = 2^-14
            d.append(Data([UInt8(bias & 0xFF), UInt8(bias >> 8)]))
            for _ in 0..<4 {
                let w = UInt32(truncatingIfNeeded: next() >> 32)
                d.append(Data([UInt8(w & 0xFF), UInt8((w >> 8) & 0xFF),
                               UInt8((w >> 16) & 0xFF), UInt8((w >> 24) & 0xFF)]))
            }
        }
        return d
    }

    /// The chain's input: the rotated normed hidden, f16. Row `t` is row 0
    /// scaled, so every row's logits keep the same ordering and the gaps.
    static func rotatedHidden(rows: Int, hidden: Int) -> [Float] {
        var v = [Float](repeating: 0, count: rows * hidden)
        for t in 0..<rows {
            let scale = 1.0 + 0.1 * Float(t)
            for j in 0..<hidden {
                v[t * hidden + j] = (Float((j % 23) - 11) * 0.125 + 0.0625) * scale
            }
        }
        return v.map { QWQuant.f16tof32Bits(QWQuant.f32tof16Bits($0)) }
    }

    /// The PACKER's rotation — not a copy of it. `fwht32OrthoBlock` is the
    /// function the packer rotates the weights with.
    static func unrotate(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
            }
        }
    }

    /// The router plane: a deterministic pseudo-RANDOM matrix, independent of
    /// the hidden. A router parallel to the hidden gives every logit the same
    /// scale — the same softmax, the same top-k — so it cannot tell the bases apart.
    static func routerPlane(experts: Int, hidden: Int, xPlain: [Float]) -> [Float] {
        var seed: UInt64 = 0x51ED270B
        var plane = [Float](repeating: 0, count: experts * hidden)
        for i in 0..<(experts * hidden) {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let u = Double((seed >> 11) & 0x1FFFFFFFFFFFFF) / Double(0x20000000000000)
            plane[i] = Float(u - 0.5)
        }
        // One global scale, so the logits' spread is the one the test wants.
        let logits = routerLogits(plane: plane, x: xPlain, experts: experts, hidden: hidden)
        let mean = logits.reduce(0, +) / Double(max(1, logits.count))
        let var0 = logits.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(1, logits.count))
        let scale = 0.8 / max(1e-6, var0.squareRoot())
        return plane.map { Float(Double($0) * scale) }
    }

    /// The router's logits for one row: `plane[e] · x`.
    static func routerLogits(plane: [Float], x: [Float], experts: Int, hidden: Int) -> [Double] {
        (0..<experts).map { e in
            var acc = 0.0
            for k in 0..<hidden { acc += Double(plane[e * hidden + k]) * Double(x[k]) }
            return acc
        }
    }

    /// The top-k the router picks, and the gap between the last pick and the
    /// first one left out — the gap is what makes the selection unambiguous.
    static func topK(_ logits: [Double], routes: Int) -> (picked: [Int], boundaryGap: Double) {
        let order = logits.indices.sorted { a, b in
            logits[a] == logits[b] ? a < b : logits[a] > logits[b]
        }
        let picked = Array(order.prefix(routes))
        let boundary = order.count > routes ? logits[order[routes - 1]] - logits[order[routes]] : 0
        return (picked, boundary)
    }

    /// Everything the chain needs: the two banks, the router, the hidden, and
    /// the banks dequantised into the standard basis for the reference.
    struct Model {
        let geometry: QWModelGeometry
        let xRot: [Float]
        let router: [Float]
        let guBank: Data
        let downBank: Data
        let guStd: [Float]
        let downStd: [Float]
        var hidden: Int { geometry.hidden }
        var intermediate: Int { geometry.moeIntermediate }
        var experts: Int { geometry.numExperts }
        var routes: Int { geometry.expertsPerTok }
    }

    static func model(rows: Int, normTopKProb: Bool) throws -> Model {
        let g = try geometry(normTopKProb: normTopKProb)
        let hidden = g.hidden, inter = g.moeIntermediate, experts = g.numExperts
        var seed: UInt64 = 0xC0FFEE
        let guBank = makeBank(rows: experts * 2 * inter, k: hidden, seed: &seed)
        let downBank = makeBank(rows: experts * hidden, k: inter, seed: &seed)
        let xRot = rotatedHidden(rows: rows, hidden: hidden)
        var xPlain = xRot
        unrotate(&xPlain)
        let router = routerPlane(experts: experts, hidden: hidden, xPlain: Array(xPlain[0..<hidden]))
        let guStd = guBank.withUnsafeBytes {
            QWQuant.Pack.dequantHQ4Slab(base: $0.baseAddress!, nRows: experts * 2 * inter, kCols: hidden)
        }
        let downStd = downBank.withUnsafeBytes {
            QWQuant.Pack.dequantHQ4Slab(base: $0.baseAddress!, nRows: experts * hidden, kCols: inter)
        }
        return Model(geometry: g, xRot: xRot, router: router, guBank: guBank, downBank: downBank,
                     guStd: guStd, downStd: downStd)
    }

    // MARK: - The chain on the GPU

    /// The chain's two outputs: the f32 result and the f16 narrowing of it.
    /// The f16 slot is the one the residual stream reads, so its bits are
    /// the layer's real output.
    struct ChainOut {
        let f32: [Float]
        let f16Bits: [UInt16]
    }

    /// Run the chain over `rows` rows. `mutate` may rewrite the scratch
    /// first; `chainGeometry` is the geometry the CHAIN reads. The buffers
    /// are the PRODUCTION slots, so a chain that wrote elsewhere is caught.
    static func runChain(model: Model, rows: Int,
                         chainGeometry: QWModelGeometry? = nil,
                         mutate: ((inout QWEngineDriver.MoEScratch) -> Void)? = nil) throws -> ChainOut {
        let geometry = model.geometry
        let hidden = model.hidden
        let intermediate = model.intermediate
        let experts = model.experts
        let routes = model.routes
        let guBytes = model.guBank.count
        let downBytes = model.downBank.count
        let arena = try QWArena(hal: hal, weights: guBytes + downBytes + experts * hidden * 4 + (1 << 20),
                                activations: 4 << 20, kv: 0, gdnStates: 0)

        let w = arena.weightsRegion()
        let guBlocks = try w.allocate(QWTensorShape([guBytes / 2], .f16))
        // The banks and the router's plane go in first: they are the
        // model's own weights, in the packer's own layout.
        arena.copyIn(guBlocks, from: model.guBank)
        let downBlocks = try w.allocate(QWTensorShape([downBytes / 2], .f16))
        arena.copyIn(downBlocks, from: model.downBank)
        let routerTV = try w.allocate(QWTensorShape([experts * hidden], .f32))
        arena.copyIn(routerTV, from: model.router.withUnsafeBufferPointer { Data(buffer: $0) })
        guard let gateUp = QWWeightView(blocks: guBlocks, n: experts * 2 * intermediate,
                                            k: hidden, isRotated: true),
              let down = QWWeightView(blocks: downBlocks, n: experts * hidden,
                                       k: intermediate, isRotated: true) else {
            throw NSError(domain: "moe", code: 1, userInfo: [NSLocalizedDescriptionKey: "the bank view is invalid"])
        }
        let moe = QWEngineDriver.MoEW(router: routerTV, gateUp: gateUp, down: down,
                                       numExperts: experts, expertsPerTok: routes)

        let a = arena.activationsRegion()
        let x16 = try a.allocate(QWTensorShape([rows * hidden], .f16))
        var bits = [UInt16](repeating: 0, count: rows * hidden)
        for i in 0..<(rows * hidden) { bits[i] = QWQuant.f32tof16Bits(model.xRot[i]) }
        // The production slots: the post-attention norm's f16 hidden in,
        // the f32 MLP slot and the f16 one out — the widths the driver
        // carves for a decode step and for a prefill chunk alike.
        arena.copyIn(x16, from: Data(bytes: bits, count: bits.count * 2))
        let dn32 = try a.allocate(QWTensorShape([rows * hidden], .f32))
        let dn16 = try a.allocate(QWTensorShape([rows * hidden], .f16))

        var scratch = try QWEngineDriver.makeMoEScratch({ n, dt in try a.allocate(QWTensorShape([n], dt)) },
                                                        rows: rows, geometry: geometry)
        mutate?(&scratch)

        guard let pair = hal.begin() else { throw XCTSkip("no command buffer") }
        let cb = pair.0
        var enc = pair.1
        do {
            try QWMoEChain.encode(enc: &enc, hal: hal, moe: moe, s: scratch,
                                  x16: x16, dn32: dn32, dn16: dn16,
                                  rows: rows, geometry: chainGeometry ?? geometry, cb: cb)
        } catch {
            enc.endEncoding()
            throw error
        }
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "the chain failed: \(String(describing: cb.error))")

        let p32 = arena.cpuPointer(of: dn32).assumingMemoryBound(to: Float.self)
        let p16 = arena.cpuPointer(of: dn16).assumingMemoryBound(to: UInt16.self)
        return ChainOut(f32: (0..<(rows * hidden)).map { p32[$0] },
                        f16Bits: (0..<(rows * hidden)).map { p16[$0] })
    }

    // MARK: - The reference

    /// The CPU reference: the router's softmax and top-k, each expert's
    /// SwiGLU and down, and the weighted sum. The two flags are the ways the
    /// chain can be WRONG: the wrong basis on either side.
    static func reference(model: Model, rows: Int, normTopK: Bool,
                          routerRotated: Bool = false, expertsRotated: Bool = false) -> [Float] {
        let hidden = model.hidden, intermediate = model.intermediate
        let experts = model.experts, routes = model.routes
        var out = [Double](repeating: 0, count: rows * hidden)
        for t in 0..<rows {
            // The standard-basis hidden: the packer's own rotation of the f16 row.
            var plain = [Float](model.xRot[(t * hidden)..<((t + 1) * hidden)])
            unrotate(&plain)
            let rotated = Array(model.xRot[(t * hidden)..<((t + 1) * hidden)])
            let routerX = routerRotated ? rotated : plain
            let expertsX = expertsRotated ? rotated : plain
            // The router's logits over the UN-ROTATED hidden.
            var logits = [Double](repeating: 0, count: experts)
            for e in 0..<experts {
                var acc = 0.0
                for k in 0..<hidden { acc += Double(model.router[e * hidden + k]) * Double(routerX[k]) }
                logits[e] = acc
            }
            // The softmax, then the top-k (ties to the lower index).
            let top = logits.max()!
            var p = logits.map { exp($0 - top) }
            let sum = p.reduce(0, +)
            p = p.map { $0 / sum }
            let order = logits.indices.sorted { a, b in
                logits[a] == logits[b] ? a < b : logits[a] > logits[b]
            }
            let picked = Array(order.prefix(routes))
            let wsum = picked.reduce(0.0) { $0 + p[$1] }
            for e in picked {
                let w = normTopK ? p[e] / wsum : p[e]
                // The expert's gate/up over the standard-basis hidden.
                var act = [Double](repeating: 0, count: intermediate)
                for i in 0..<intermediate {
                    var g = 0.0, u = 0.0
                    for k in 0..<hidden {
                        g += Double(model.guStd[(e * 2 * intermediate + i) * hidden + k]) * Double(expertsX[k])
                        u += Double(model.guStd[(e * 2 * intermediate + intermediate + i) * hidden + k]) * Double(expertsX[k])
                    }
                    act[i] = (g / (1 + exp(-g))) * u
                }
                // The expert's down.
                for j in 0..<hidden {
                    var acc = 0.0
                    for i in 0..<intermediate {
                        acc += Double(model.downStd[(e * hidden + j) * intermediate + i]) * act[i]
                    }
                    out[t * hidden + j] += w * acc
                }
            }
        }
        return out.map { Float($0) }
    }

    /// The worst gap between two vectors, relative to the reference's own
    /// scale — a per-element ratio would blow up where the reference is zero.
    static func gap(_ got: [Float], _ want: [Float]) -> Double {
        let scale = max(1e-6, want.map { abs(Double($0)) }.max() ?? 1e-6)
        var worst = 0.0
        for i in 0..<got.count {
            worst = max(worst, abs(Double(got[i]) - Double(want[i])) / scale)
        }
        return worst
    }

    // MARK: - The chain

    /// The chain's output is the reference's, to within the f32 arithmetic.
    func testTheRoutedLayerMatchesTheCPUReference() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "Routed MoE prefill (rows=16) requires Metal 4 tensor hardware")
        for rows in [1, 3, 16] {
            let model = try Self.model(rows: rows, normTopKProb: true)
            let got = try Self.runChain(model: model, rows: rows)
            let want = Self.reference(model: model, rows: rows, normTopK: true)
            let g = Self.gap(got.f32, want)
            let tol = rows > 8 ? 5e-3 : 1e-4
            XCTAssertLessThan(g, tol, "rows \(rows): the chain differs from the reference by \(g)")
        }
    }

    /// The chain's LAST step narrows the f32 result into the f16 slot the
    /// residual stream reads. The oracle reads that slot back: the bits must
    /// BE the f32 result's f16 rounding.
    func testTheChainNarrowsItsResultIntoTheF16Slot() throws {
        let model = try Self.model(rows: 1, normTopKProb: true)
        let out = try Self.runChain(model: model, rows: 1)
        XCTAssertEqual(out.f16Bits.count, out.f32.count)
        var lossy = 0
        for i in out.f32.indices {
            let want = QWQuant.f32tof16Bits(out.f32[i])
            XCTAssertEqual(out.f16Bits[i], want,
                           "element \(i): the f16 slot is not the f32 result's narrowing")
            if want != out.f32[i].bitPattern { lossy += 1 }
        }
        XCTAssertGreaterThan(lossy, 0, "the narrowing must be lossy, or the f16 slot was never written")
    }

    /// The chain reads the model's own top-k normalisation flag — never a
    /// constant. The same model with the flag off must produce the
    /// un-normalised weighted sum.
    func testTheTopKNormalisationIsTheModelsOwnFlag() throws {
        let model = try Self.model(rows: 1, normTopKProb: false)
        let got = try Self.runChain(model: model, rows: 1)
        let want = Self.reference(model: model, rows: 1, normTopK: false)
        XCTAssertLessThan(Self.gap(got.f32, want), 1e-4, "the flag-off chain must match the flag-off reference")

        // The same weights with the flag ON must give a different answer.
        let onModel = try Self.model(rows: 1, normTopKProb: true)
        let on = try Self.runChain(model: onModel, rows: 1)
        XCTAssertGreaterThan(Self.gap(on.f32, got.f32), 1e-3, "the flag must change the result")
    }

    /// The fixture must be able to FAIL: the top-k it hands the router is
    /// unambiguous, and the two bases pick DIFFERENT experts. A fixture whose
    /// two readings agree cannot tell a chain that reads the wrong one.
    func testTheFixtureCanTellTheTwoBasesApart() throws {
        let model = try Self.model(rows: 1, normTopKProb: true)
        let hidden = model.hidden
        var plain = [Float](model.xRot[0..<hidden])
        Self.unrotate(&plain)
        let rotated = Array(model.xRot[0..<hidden])
        let plainLogits = Self.routerLogits(plane: model.router, x: plain,
                                            experts: model.experts, hidden: hidden)
        let rotatedLogits = Self.routerLogits(plane: model.router, x: rotated,
                                              experts: model.experts, hidden: hidden)
        let plainTop = Self.topK(plainLogits, routes: model.routes)
        let rotatedTop = Self.topK(rotatedLogits, routes: model.routes)
        XCTAssertGreaterThan(plainTop.boundaryGap, 1e-3,
                             "the top-k must be unambiguous: a near-tie would make the test flaky")
        XCTAssertNotEqual(plainTop.picked, rotatedTop.picked,
                          "the two bases must pick different experts, or the test cannot see the swap")
        XCTAssertGreaterThan(Self.gap(rotatedLogits.map { Float($0) },
                                       plainLogits.map { Float($0) }), 0.1,
                             "the two bases must give different logits")
    }

    /// Each stage, broken on purpose, must be visible to the reference. The
    /// two BASES are broken in the reference, not on the GPU — the chain
    /// writes both bridge outputs itself.
    func testTheChainNoticesEachBrokenStage() throws {
        let model = try Self.model(rows: 1, normTopKProb: true)
        let got = try Self.runChain(model: model, rows: 1)
        let want = Self.reference(model: model, rows: 1, normTopK: true)
        XCTAssertLessThan(Self.gap(got.f32, want), 1e-4)

        // The router reading the ROTATED hidden (the packer rotates the
        // experts' input dimension, never the router's raw plane).
        let routerRotated = Self.reference(model: model, rows: 1, normTopK: true, routerRotated: true)
        XCTAssertGreaterThan(Self.gap(got.f32, routerRotated), 1e-3,
                             "a router that reads the rotated hidden must be caught")

        // The experts reading the UN-ROTATED hidden.
        let expertsRotated = Self.reference(model: model, rows: 1, normTopK: true, expertsRotated: true)
        XCTAssertGreaterThan(Self.gap(got.f32, expertsRotated), 1e-3,
                             "experts that read the un-rotated hidden must be caught")

        // Both wrong at once.
        let bothRotated = Self.reference(model: model, rows: 1, normTopK: true,
                                         routerRotated: true, expertsRotated: true)
        XCTAssertGreaterThan(Self.gap(got.f32, bothRotated), 1e-3)

        // The shared-expert term not being zero: the gate is on and the
        // shared buffer holds something.
        let shared = try Self.runChain(model: model, rows: 1) { s in
            let p = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            p[0] = 1.0
            let q = s.shared.region.pointer.advanced(by: s.shared.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<(s.shared.shape.elementCount) { q[i] = 4.0 }
        }
        XCTAssertGreaterThan(Self.gap(shared.f32, want), 1e-3,
                             "a non-zero shared-expert term must be caught")
    }

    // MARK: - The engine's side

    /// The plan's `.moe` layer key has a kernel: the sparse layer is served,
    /// not refused.
    func testTheSparseLayerKeyHasAKernel() {
        let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: false)
        XCTAssertNotNil(QWLayerKernelRegistry.kernel(for: key))
    }

    /// The scratch grows only for a model that HAS a sparse layer: the 27B's
    /// activations region must not change.
    func testTheScratchGrowsOnlyForASparseModel() throws {
        XCTAssertEqual(QWEngineDriver.moeBytesPerToken(geometry: .qwen38_27b), 0,
                       "the 27B has no MoE layer")
        let g = try Self.geometry(normTopKProb: true)
        XCTAssertGreaterThan(QWEngineDriver.moeBytesPerToken(geometry: g), 0)
        // The dense prefill law is unchanged for the 27B.
        XCTAssertEqual(QWEngineDriver.prefillBytesPerToken(geometry: .qwen38_27b), 748_096)
    }

    /// The scratch's byte law and its allocation are ONE number: a budget
    /// that counts a different footprint than the one carved is a budget
    /// that lies.
    func testTheScratchsByteLawIsItsAllocation() throws {
        let g = try Self.geometry(normTopKProb: true)
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 4 << 20, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        for rows in [1, 3] {
            let scratch = try QWEngineDriver.makeMoEScratch({ n, dt in
                try a.allocate(QWTensorShape([n], dt))
            }, rows: rows, geometry: g)
            var carved = 0
            for t in [scratch.xRot32, scratch.xPlain32, scratch.logits, scratch.idx,
                      scratch.wts, scratch.gu32, scratch.act32, scratch.shared, scratch.gate,
                      scratch.dnAcc] {
                carved += t.shape.totalBytes
            }
            XCTAssertEqual(carved, QWEngineDriver.moeBytesPerToken(geometry: g) * rows,
                           "rows \(rows): the law and the carve disagree")
            // The scratch's own shapes: the activation buffer is [T·topK, I].
            XCTAssertEqual(scratch.gu32.shape.elementCount, rows * g.expertsPerTok * 2 * g.moeIntermediate)
            XCTAssertEqual(scratch.act32.shape.elementCount, rows * g.expertsPerTok * g.moeIntermediate)
            XCTAssertEqual(scratch.logits.shape.elementCount, rows * g.numExperts)
            XCTAssertEqual(scratch.idx.shape.elementCount, rows * g.expertsPerTok)
            // The shared term is zero — the gate is zero and the shared
            // buffer is zeroed, so `g · shared` is zero.
            let gate = arena.cpuPointer(of: scratch.gate).assumingMemoryBound(to: Float.self)
            XCTAssertEqual(gate[0], 0)
            let shared = arena.cpuPointer(of: scratch.shared).assumingMemoryBound(to: Float.self)
            for i in 0..<min(64, scratch.shared.shape.elementCount) { XCTAssertEqual(shared[i], 0) }
        }
    }

    /// Decode's final sum is in router order, independent of which route was
    /// resident when its down projection ran. This cancellation is the
    /// floating-point case that changes under resident-first accumulation.
    func testDecodeRouteCombineUsesRouterOrderAndAddsSharedLast() throws {
        let geometry = try Self.geometry(normTopKProb: true)
        let hidden = geometry.hidden
        let routes = geometry.expertsPerTok
        let arena = try QWArena(hal: Self.hal, weights: 0, activations: 1 << 20, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let routeDown = try a.allocate(QWTensorShape([routes * hidden], .f32))
        let weights = try a.allocate(QWTensorShape([routes], .f32))
        let shared = try a.allocate(QWTensorShape([hidden], .f32))
        let gate = try a.allocate(QWTensorShape([1], .f32))
        let out = try a.allocate(QWTensorShape([hidden], .f32))
        let out16 = try a.allocate(QWTensorShape([hidden], .f16))

        let routePtr = arena.cpuPointer(of: routeDown).assumingMemoryBound(to: Float.self)
        let weightPtr = arena.cpuPointer(of: weights).assumingMemoryBound(to: Float.self)
        let sharedPtr = arena.cpuPointer(of: shared).assumingMemoryBound(to: Float.self)
        let gatePtr = arena.cpuPointer(of: gate).assumingMemoryBound(to: Float.self)
        for i in 0..<(routes * hidden) { routePtr[i] = 0 }
        for i in 0..<routes { weightPtr[i] = 1 }
        weightPtr[routes - 1] = 0
        for i in 0..<hidden {
            routePtr[i] = 100_000_000
            routePtr[hidden + i] = 1
            routePtr[2 * hidden + i] = -100_000_000
            routePtr[(routes - 1) * hidden + i] = .nan
            sharedPtr[i] = 0.25
        }
        gatePtr[0] = 1

        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeMoEDownReduceCombineF32(enc, hal: Self.hal, dnAcc: routeDown, invPerm: nil,
                                                 weights: weights, shared: shared, gate: gate, out: out,
                                                 hidden: hidden, routes: routes, tokens: 1, out16: out16)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "the route reduction failed: \(String(describing: cb.error))")

        let outPtr = arena.cpuPointer(of: out).assumingMemoryBound(to: Float.self)
        for i in 0..<hidden {
            var expected: Float = 0
            for route in 0..<routes where weightPtr[route] != 0 {
                expected += weightPtr[route] * routePtr[route * hidden + i]
            }
            expected += gatePtr[0] * sharedPtr[i]
            XCTAssertEqual(outPtr[i].bitPattern, expected.bitPattern, "hidden channel \(i)")
        }
        let oldResidentFirst = Float(100_000_000) + Float(-100_000_000) + Float(0.25) + Float(1)
        XCTAssertEqual(oldResidentFirst, 1.25)
        XCTAssertEqual(outPtr[0], 0.25)
    }

    /// The sparse module's doors are the per-pair chain's six: the chain's
    /// own kernels and the router's GEMM. The dense standard families route
    /// nothing, so they declare none of them.
    func testTheSparseDoorsAreThePerPairChain() {
        let sparse = Set(Qwen3MoEModule.requiredDoors)
        let dense = Set(StandardModule.requiredDoors)
        XCTAssertEqual(Set(Qwen3MoEModule.moeDoors),
                       ["qwg_router_gemm", "qwg_moe_bridge_f16_to_f32", "qwg_moe_route",
                        "qwg_moe_route_decode", "qwg_moe_gateup_q4g_f32_r1",
                        "qwg_moe_down_shared_combine_q4g_f32",
                        "qwg_moe_group_pairs_f32", "qwg_moe_nax_gemm",
                        "qwg_moe_nax_split_f32", "qwg_moe_down_reduce_combine_f32"])
        for door in Qwen3MoEModule.moeDoors {
            XCTAssertTrue(sparse.contains(door), door)
            XCTAssertFalse(dense.contains(door), door)
        }
        XCTAssertTrue(sparse.isSubset(of: Set(DenseModule.requiredDoors)))
        for grouped in ["qwg_moe_gateup_group_f32"] {
            XCTAssertFalse(sparse.contains(grouped), grouped)
        }
    }

    /// The config laws: what the chain cannot serve is refused by name, and
    /// what it can serve is honoured.
    func testTheConfigRefusesWhatTheChainCannotServe() throws {
        let sparse: [String: Any] = [
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "num_experts": 32,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let dense: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 64,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        func decode(_ base: [String: Any], _ extra: [String: Any]) throws -> QWModelDescriptor {
            var config = base
            for (k, v) in extra { config[k] = v }
            let data = try JSONSerialization.data(withJSONObject: config)
            return try QWModelDescriptor.decode(configJSON: data)
        }
        // The experts' width is the model's own: the sparse family without
        // it is refused, not served with empty banks.
        var noWidth = sparse
        noWidth.removeValue(forKey: "moe_intermediate_size")
        XCTAssertThrowsError(try decode(noWidth, [:])) { e in
            XCTAssertTrue("\(e)".contains("moe_intermediate_size"), "\(e)")
        }
        XCTAssertEqual(try decode(sparse, [:]).moeIntermediate, 64)
        // An experts' width on a DENSE family is refused by name: the model
        // has no experts to give the width to.
        XCTAssertThrowsError(try decode(dense, ["moe_intermediate_size": 64])) { e in
            XCTAssertTrue("\(e)".contains("moe_intermediate_size"), "\(e)")
        }
        // The shared expert: the standard path has none, so its term is
        // zeroed — a checkpoint with one is refused by name. A width of zero
        // IS "no shared expert", so it is honoured.
        XCTAssertThrowsError(try decode(sparse, ["shared_expert_intermediate_size": 128])) { e in
            XCTAssertTrue("\(e)".contains("shared_expert_intermediate_size"), "\(e)")
        }
        XCTAssertThrowsError(try decode(dense, ["shared_expert_intermediate_size": 128])) { e in
            XCTAssertTrue("\(e)".contains("shared_expert_intermediate_size"), "\(e)")
        }
        XCTAssertEqual(try decode(sparse, ["shared_expert_intermediate_size": 0]).sharedExpertIntermediate, 0)
        // The router's law: the experts tile 32 at a time and hold 16 tiles.
        // Zero experts is outside the law — the route would select nothing.
        XCTAssertThrowsError(try decode(sparse, ["num_experts": 0])) { e in
            XCTAssertTrue("\(e)".contains("num_experts"), "\(e)")
        }
        XCTAssertThrowsError(try decode(sparse, ["num_experts": 24]))
        XCTAssertThrowsError(try decode(sparse, ["num_experts": 544]))
        XCTAssertThrowsError(try decode(sparse, ["num_experts_per_tok": 17]))
        // The model's own top-k normalisation flag is honoured.
        XCTAssertTrue(try decode(sparse, ["norm_topk_prob": true]).normTopKProb)
        XCTAssertFalse(try decode(sparse, [:]).normTopKProb)
    }
}
