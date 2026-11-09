// QWStandardMoETests.swift — The standard MoE path (Task 4.4).
//
// The weight side lands first: a standard `qwen3_moe` checkpoint's router
// and expert banks load into ONE contiguous view each (C119). The compute
// side — the routed layer's chain — is Task 4.4c, and it is now here: the
// `.moe` layer key has its own kernel, and the engine never falls back to
// the dense path for a sparse layer.

import XCTest
import Foundation
import Metal
@testable import QWEngine
@testable import QWCore
@testable import QWQuant
@testable import QWHAL
@testable import QWAlloc
@testable import QWGEMM

final class QWStandardMoETests: XCTestCase {
    /// One synthetic plane: a verbatim bf16 vector or an hq4 matrix whose
    /// bytes are a single tag, so a loaded plane says where it came from.
    private enum Plane {
        case raw([Float])
        case hq4(n: Int, k: Int, tag: UInt8)
    }

    // MARK: - the shape

    /// The fixture is a sparse MoE geometry: 32 experts (the router's law
    /// is `experts % 32 == 0`), top-8, a 256-wide expert intermediate.
    func testTheMoEFixtureCarriesTheExperts() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        XCTAssertEqual(geometry.numExperts, 32)
        XCTAssertEqual(geometry.expertsPerTok, 8)
        XCTAssertEqual(geometry.moeIntermediate, 64)
        XCTAssertEqual(geometry.descriptor.family, .qwen3MoE)
    }

    /// The plan names the layer's MLP family: a model with experts gets a
    /// `.moe` key, a dense model a `.dense` one.
    func testThePlanNamesTheMoEKey() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "stub",
                                    geometryDigest: "stub", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        XCTAssertEqual(plan.layers.count, geometry.numLayers)
        for layer in plan.layers {
            XCTAssertEqual(layer.kernelKey.mlp, .moe)
            XCTAssertEqual(layer.kernelKey.attention, .fullAttention)
        }
        let dense = try QWEnginePlan(geometry: try QWModelGeometry.stubTinyLlamaGeometry(),
                                     modelId: "stub", geometryDigest: "stub",
                                     prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(dense.layers[0].kernelKey.mlp, .dense)
    }

    /// The compute side has landed: the `.moe` key HAS a kernel, and the
    /// plan resolves to it — the engine never falls back to the dense path.
    func testTheMoEKeyHasAKernel() throws {
        let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: false)
        XCTAssertNotNil(QWLayerKernelRegistry.kernel(for: key))
        let geometry = try QWModelGeometry.stubPureMoE()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "stub",
                                    geometryDigest: "stub", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let kernels = try QWLayerKernelRegistry.resolve(plan: plan)
        XCTAssertEqual(kernels.count, geometry.numLayers)
    }

    /// The plan and the weights must agree about the MLP family too — a
    /// plan that says `.dense` against MoE weights is a different model.
    func testTheCrossCheckComparesTheMLPFamily() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        let (layers, _) = try loadMoE(geometry: geometry)
        let dense = try QWEnginePlan(geometry: try QWModelGeometry.stubTinyLlamaGeometry(),
                                     modelId: "stub", geometryDigest: "stub",
                                     prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertThrowsError(try QWLayerKernelRegistry.resolve(layers: layers, plan: dense)) { error in
            XCTAssertTrue("\(error)".contains("moe"), "the refusal names the kind: \(error)")
        }
    }

    // MARK: - the weights

    /// The loader stacks a standard checkpoint's per-expert planes into ONE
    /// contiguous bank: the gate/up bank is `[E·2I, K]` — every expert's gate
    /// rows then its up rows — and the down bank is `[E·H, I]` (C119).
    func testTheExpertBanksAreOneContiguousViewEach() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        let (layers, _) = try loadMoE(geometry: geometry)
        XCTAssertEqual(layers.count, geometry.numLayers)
        let moe = try XCTUnwrap(layers[0].moe)
        XCTAssertEqual(layers[0].kind, .moe)
        XCTAssertEqual(moe.numExperts, 32)
        XCTAssertEqual(moe.expertsPerTok, 8)
        XCTAssertEqual(moe.gateUp.n, 32 * 2 * geometry.moeIntermediate)
        XCTAssertEqual(moe.gateUp.k, geometry.hidden)
        XCTAssertEqual(moe.down.n, 32 * geometry.hidden)
        XCTAssertEqual(moe.down.k, geometry.moeIntermediate)
        // The attention is the same full attention as a dense layer's.
        XCTAssertNotNil(layers[0].attn)
        XCTAssertNil(layers[0].gdn)
    }

    /// The experts are stacked IN ORDER: expert e's gate rows sit at
    /// `e·2I`, its up rows at `e·2I + I`, and its down slab at `e·H`.
    /// Every expert carries a distinct tag, so the bytes say which is which.
    func testTheExpertsAreStackedInOrder() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        let (layers, _) = try loadMoE(geometry: geometry)
        let moe = try XCTUnwrap(layers[0].moe)
        let I = geometry.moeIntermediate, H = geometry.hidden
        let g = H / 32
        let gateBytes = I * g * 20
        let upBytes = gateBytes
        let downBytes = H * (I / 32) * 20
        for e in 0..<4 {
            let gateTag = UInt8(0x10 + e)
            let upTag = UInt8(0x40 + e)
            let downTag = UInt8(0x70 + e)
            XCTAssertEqual(bytes(moe.gateUp.blocks, at: e * 2 * I * g * 20, count: 1), [gateTag],
                           "expert \(e)'s gate rows")
            XCTAssertEqual(bytes(moe.gateUp.blocks, at: (e * 2 * I + I) * g * 20, count: 1), [upTag],
                           "expert \(e)'s up rows")
            XCTAssertEqual(bytes(moe.down.blocks, at: e * H * (I / 32) * 20, count: 1), [downTag],
                           "expert \(e)'s down slab")
            XCTAssertEqual(gateBytes, upBytes)
            XCTAssertGreaterThan(downBytes, 0)
        }
    }

    /// The router is the pack's RAW bf16 plane — the packer copies it
    /// verbatim, not quantised and not rotated — so the loader widens it
    /// losslessly (a bf16 → f32 → bf16 round trip is exact).
    func testTheRouterIsTheRawPlane() throws {
        let geometry = try QWModelGeometry.stubPureMoE()
        let (layers, _) = try loadMoE(geometry: geometry)
        let moe = try XCTUnwrap(layers[0].moe)
        XCTAssertEqual(moe.router.shape.elementCount, 32 * geometry.hidden)
        XCTAssertEqual(moe.router.shape.dtype, .f32)
        // The first router row carries the fixture's sentinel values.
        let values = f32Values(moe.router, count: 4)
        XCTAssertEqual(values[0], 1.5, accuracy: 1e-6)
        XCTAssertEqual(values[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(values[2], 2.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], 1.0, accuracy: 1e-6)
    }

    // MARK: - the loader

    /// The pack's names, spelled the way a standard MoE checkpoint spells
    /// them: the router is `mlp.gate.weight`, and each expert carries its
    /// own gate/up/down planes.
    private enum Pack {
        static let ln1 = "model.layers.0.input_layernorm.weight"
        static let ln2 = "model.layers.0.post_attention_layernorm.weight"
        static let router = "model.layers.0.mlp.gate.weight"
        static let q = "model.layers.0.self_attn.q_proj.weight"
        static let k = "model.layers.0.self_attn.k_proj.weight"
        static let v = "model.layers.0.self_attn.v_proj.weight"
        static let o = "model.layers.0.self_attn.o_proj.weight"
        static let qNorm = "model.layers.0.self_attn.q_norm.weight"
        static let kNorm = "model.layers.0.self_attn.k_norm.weight"
        static let finalNorm = "model.norm.weight"
        static let head = "lm_head.weight"
        static func expert(_ e: Int, _ w: String) -> String {
            "model.layers.0.mlp.experts.\(e).\(w).weight"
        }
    }

    private enum Tag {
        static let q: UInt8 = 0xC3, k: UInt8 = 0xD4, v: UInt8 = 0xE5, o: UInt8 = 0xF6
        static let head: UInt8 = 0x11
    }

    private static let gammas: [Float] = [1.5, 0.5, 2.0, 1.0, 0.25, 1.25]

    private func gammaVector(_ count: Int) -> [Float] {
        (0..<count).map { Self.gammas[$0 % Self.gammas.count] }
    }

    /// Writes a QWP1 pack: the planes at their file offsets, the header at
    /// the end (the reader takes the header length from bytes 4..12).
    private func writePack(at path: String, planes: [String: Plane]) throws {
        var payload = Data()
        var entries: [String: [String: Any]] = [:]
        var offset = 12
        for (name, plane) in planes {
            let aligned = (offset + 255) & ~255
            payload.append(Data(repeating: 0, count: aligned - offset))
            offset = aligned
            switch plane {
            case .raw(let values):
                var bytes = Data(capacity: values.count * 2)
                for v in values {
                    let bf = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
                    withUnsafeBytes(of: bf.littleEndian) { bytes.append(contentsOf: $0) }
                }
                entries[name] = ["shape": [values.count], "raw_off": offset, "kind": "bf16"]
                payload.append(bytes)
                offset += bytes.count
            case .hq4(let n, let k, let tag):
                let bytes = n * (k / 32) * 20
                entries[name] = ["shape": [n, k], "hq4_off": offset]
                payload.append(Data(repeating: tag, count: bytes))
                offset += bytes
            }
        }
        let header = try JSONSerialization.data(withJSONObject: [
            "version": 2,
            "provenance": ["quant": "synthetic-test"],
            "tensors": entries,
        ])
        var file = Data()
        file.append(Data("QWP1".utf8))
        withUnsafeBytes(of: UInt64(header.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(payload)
        withUnsafeBytes(of: UInt64(payload.count).littleEndian) { file.append(contentsOf: $0) }
        file.append(header)
        try Data(file).write(to: URL(fileURLWithPath: path))
    }

    /// The MoE checkpoint's planes: the router (raw), every expert's three
    /// planes (each with its OWN tag, so the stacking is checkable), and the
    /// attention + norms the layer shares with a dense one.
    private func moePlanes(_ geometry: QWModelGeometry) -> [String: Plane] {
        let h = geometry.hidden, v = geometry.vocab, I = geometry.moeIntermediate
        let kv = geometry.kvHeads * geometry.headDim
        var planes: [String: Plane] = [
            Pack.ln1: .raw(gammaVector(h)),
            Pack.ln2: .raw(gammaVector(h)),
            Pack.router: .raw(gammaVector(geometry.numExperts * h)),
            Pack.q: .hq4(n: geometry.qProjOut, k: h, tag: Tag.q),
            Pack.k: .hq4(n: kv, k: h, tag: Tag.k),
            Pack.v: .hq4(n: kv, k: h, tag: Tag.v),
            Pack.o: .hq4(n: h, k: geometry.qHeads * geometry.headDim, tag: Tag.o),
            // A qwen3_moe carries the per-head q/k norms (Task 4.5 applies
            // them; the loader reads them here).
            Pack.qNorm: .raw(gammaVector(geometry.headDim)),
            Pack.kNorm: .raw(gammaVector(geometry.headDim)),
            Pack.finalNorm: .raw(gammaVector(h)),
            Pack.head: .hq4(n: v, k: h, tag: Tag.head),
        ]
        for e in 0..<geometry.numExperts {
            planes[Pack.expert(e, "gate_proj")] = .hq4(n: I, k: h, tag: UInt8(0x10 + e))
            planes[Pack.expert(e, "up_proj")] = .hq4(n: I, k: h, tag: UInt8(0x40 + e))
            planes[Pack.expert(e, "down_proj")] = .hq4(n: h, k: I, tag: UInt8(0x70 + e))
        }
        return planes
    }

    /// A loader over a synthetic MoE pack, driven through the engine's own
    /// order: the head, then the decoder's weights.
    private func loadMoE(geometry: QWModelGeometry) throws -> (
        layers: [QWEngineDriver.LayerW], finalNorm: QWTensorValue
    ) {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-moe-load-\(UUID().uuidString).qwp").path
        try writePack(at: path, planes: moePlanes(geometry))
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
        let pack = try QWQuant.Pack(url: URL(fileURLWithPath: path),
                                    descriptor: geometry.descriptor,
                                    skipIdentityCheck: true)
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 24, activations: 1 << 20,
                                kv: 1 << 16, gdnStates: 1 << 16)
        let plan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: 1,
                                               mtpDrafts: 0, headQ4g: true)
        var L = QWEngineDriver.WeightLoader(pk: pack, hl: hal,
                                            psoCvt: try hal.pipeline("qwg_bf16_to_f16"),
                                            wsub: arena.weightsRegion(), plan: plan,
                                            geometry: geometry)
        _ = try L.q4(Pack.head, n: geometry.vocab, k: geometry.hidden)
        return try L.loadDecoderWeights(geometry: geometry)
    }

    private func bytes(_ t: QWTensorValue, at offset: Int, count: Int) -> [UInt8] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset + offset)
        return [UInt8](UnsafeRawBufferPointer(start: p, count: count))
    }

    private func f32Values(_ t: QWTensorValue, count: Int) -> [Float] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            .assumingMemoryBound(to: Float.self)
        return (0..<count).map { p[$0] }
    }
}
