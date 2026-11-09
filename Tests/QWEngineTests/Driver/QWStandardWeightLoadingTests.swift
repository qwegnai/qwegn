// QWStandardWeightLoadingTests.swift — the standard path's weight loader:
// the `1 + w` gamma law and the canonical-prefix resolution, on a
// synthetic pack spelled the way a standard HF checkpoint spells it.
// The engine asks for `model.language_model.…`; this pack answers
// `model.…`, so every group has to resolve to find its bytes.
import Foundation
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
@testable import QWEngine

final class QWStandardWeightLoadingTests: XCTestCase {
    /// One synthetic plane: a verbatim bf16 vector or an hq4 matrix whose
    /// bytes are a single tag, so a loaded plane says where it came from.
    private enum Plane {
        case raw([Float])
        case hq4(n: Int, k: Int, tag: UInt8)
    }

    /// The pack's names, spelled the way a standard checkpoint spells them.
    private enum Pack {
        static let ln1 = "model.layers.0.input_layernorm.weight"
        static let ln2 = "model.layers.0.post_attention_layernorm.weight"
        static let gate = "model.layers.0.mlp.gate_proj.weight"
        static let up = "model.layers.0.mlp.up_proj.weight"
        static let down = "model.layers.0.mlp.down_proj.weight"
        static let q = "model.layers.0.self_attn.q_proj.weight"
        static let k = "model.layers.0.self_attn.k_proj.weight"
        static let v = "model.layers.0.self_attn.v_proj.weight"
        static let o = "model.layers.0.self_attn.o_proj.weight"
        static let qNorm = "model.layers.0.self_attn.q_norm.weight"
        static let kNorm = "model.layers.0.self_attn.k_norm.weight"
        static let finalNorm = "model.norm.weight"
        static let head = "lm_head.weight"
    }

    private enum Tag {
        static let gate: UInt8 = 0xA1, up: UInt8 = 0xB2, q: UInt8 = 0xC3
        static let k: UInt8 = 0xD4, v: UInt8 = 0xE5, o: UInt8 = 0xF6
        static let down: UInt8 = 0x77, head: UInt8 = 0x11
    }

    /// The gammas a standard checkpoint stores: plain γ, exactly
    /// representable in bf16 so the round trip is exact.
    private static let gammas: [Float] = [1.5, 0.5, 2.0, 1.0, 0.25, 1.25]

    private func gammaVector(_ count: Int) -> [Float] {
        (0..<count).map { Self.gammas[$0 % Self.gammas.count] }
    }

    // MARK: - the pack

    /// Writes a QWP1 pack: the planes at their file offsets, the header at
    /// the end. The reader takes the header length from bytes 4..12 and
    /// reads the header back from the file's tail.
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
        file.append(header)
        try file.write(to: URL(fileURLWithPath: path))
    }

    /// Every tensor a one-layer llama needs.
    private func standardPlanes(_ geometry: QWModelGeometry) -> [String: Plane] {
        let h = geometry.hidden, i = geometry.intermediate, v = geometry.vocab
        let kv = geometry.kvHeads * geometry.headDim
        return [
            Pack.ln1: .raw(gammaVector(h)),
            Pack.ln2: .raw(gammaVector(h)),
            Pack.gate: .hq4(n: i, k: h, tag: Tag.gate),
            Pack.up: .hq4(n: i, k: h, tag: Tag.up),
            Pack.down: .hq4(n: h, k: i, tag: Tag.down),
            Pack.q: .hq4(n: geometry.qProjOut, k: h, tag: Tag.q),
            Pack.k: .hq4(n: kv, k: h, tag: Tag.k),
            Pack.v: .hq4(n: kv, k: h, tag: Tag.v),
            Pack.o: .hq4(n: h, k: geometry.qHeads * geometry.headDim, tag: Tag.o),
            Pack.finalNorm: .raw(gammaVector(h)),
            Pack.head: .hq4(n: v, k: h, tag: Tag.head),
        ]
    }

    // MARK: - the loader

    /// A loader over a synthetic pack. The plan's allocation order is
    /// strict, so each test drives the loader in the engine's own order.
    private func makeLoader(geometry: QWModelGeometry,
                            planes: [String: Plane]) throws -> QWEngineDriver.WeightLoader {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-standard-load-\(UUID().uuidString).qwp").path
        try writePack(at: path, planes: planes)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
        let pack = try QWQuant.Pack(url: URL(fileURLWithPath: path),
                                    descriptor: geometry.descriptor,
                                    skipIdentityCheck: true)
        let hal = try QWHal(geometry: .qwen38_27b)
        let arena = try QWArena(hal: hal, weights: 1 << 20, activations: 1 << 20,
                                kv: 1 << 16, gdnStates: 1 << 16)
        let plan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: 1,
                                               mtpDrafts: 0, headQ4g: true)
        return QWEngineDriver.WeightLoader(pk: pack, hl: hal,
                                           psoCvt: try hal.pipeline("qwg_bf16_to_f16"),
                                           wsub: arena.weightsRegion(), plan: plan,
                                           geometry: geometry)
    }

    /// The loader driven through the engine's own order: the head, then the
    /// decoder's weights — every layer's and the model's final norm.
    private func loadStandard(geometry: QWModelGeometry) throws -> (
        layers: [QWEngineDriver.LayerW], finalNorm: QWTensorValue
    ) {
        var L = try makeLoader(geometry: geometry, planes: standardPlanes(geometry))
        _ = try L.q4(Pack.head, n: geometry.vocab, k: geometry.hidden)
        return try L.loadDecoderWeights(geometry: geometry)
    }

    private func bytes(_ t: QWTensorValue, count: Int) -> [UInt8] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
        return [UInt8](UnsafeRawBufferPointer(start: p, count: count))
    }

    private func f16Values(_ t: QWTensorValue, count: Int) -> [Float] {
        let p = t.region.buffer.contents().advanced(by: t.region.base + t.offset)
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(Float16(bitPattern: p[$0])) }
    }

    // MARK: - the tests

    /// Every weight group resolves: the layer norms, the MLP gate/up, the
    /// q/k/v projections, the output projection and the final norm. The
    /// pack answers `model.…`; the engine asks for `model.language_model.…`.
    func testStandardPrefixedPackResolvesEveryWeightGroup() throws {
        let geometry = try QWModelGeometry.stubTinyLlamaGeometry()
        let (layers, _) = try loadStandard(geometry: geometry)
        XCTAssertEqual(layers.count, 1)
        XCTAssertEqual(layers[0].kind, .fullAttention)
        XCTAssertNotNil(layers[0].attn)
        XCTAssertEqual(layers[0].attn?.q.n, geometry.qProjOut)
        XCTAssertEqual(layers[0].attn?.k.n, geometry.kvHeads * geometry.headDim)
        XCTAssertEqual(layers[0].attn?.o.n, geometry.hidden)
    }

    /// The MLP gate/up are concatenated in order, from the resolved names.
    func testTheGateUpConcatenationResolvesBothNames() throws {
        let geometry = try QWModelGeometry.stubTinyLlamaGeometry()
        let (layers, _) = try loadStandard(geometry: geometry)
        let half = geometry.intermediate * (geometry.hidden / 32) * 20
        let got = bytes(layers[0].gu.view.blocks, count: half * 2)
        XCTAssertTrue(got.prefix(half).allSatisfy { $0 == Tag.gate },
                      "the gate plane must come first, whole")
        XCTAssertTrue(got.suffix(half).allSatisfy { $0 == Tag.up },
                      "the up plane must follow it, whole")
    }

    /// The q/k/v projections are concatenated in order, from the resolved names.
    func testTheQKVConcatenationResolvesAllThreeNames() throws {
        let geometry = try QWModelGeometry.stubTinyLlamaGeometry()
        let (layers, _) = try loadStandard(geometry: geometry)
        let attn = try XCTUnwrap(layers[0].attn)
        let qBytes = geometry.qProjOut * (geometry.hidden / 32) * 20
        let kvBytes = geometry.kvHeads * geometry.headDim * (geometry.hidden / 32) * 20
        let got = bytes(attn.qkvProj.blocks, count: qBytes + kvBytes * 2)
        XCTAssertTrue(got.prefix(qBytes).allSatisfy { $0 == Tag.q }, "q comes first")
        XCTAssertTrue(got.dropFirst(qBytes).prefix(kvBytes).allSatisfy { $0 == Tag.k }, "k follows")
        XCTAssertTrue(got.suffix(kvBytes).allSatisfy { $0 == Tag.v }, "v comes last")
    }

    /// The model's final norm is a `1 + w` norm gamma like every other, so
    /// a standard checkpoint's plain γ is stored as γ − 1.
    func testTheFinalNormGammaIsStoredAsGammaMinusOne() throws {
        let geometry = try QWModelGeometry.stubTinyLlamaGeometry()
        let (_, finalNorm) = try loadStandard(geometry: geometry)
        let want = gammaVector(geometry.hidden).map { $0 - 1.0 }
        XCTAssertEqual(f16Values(finalNorm, count: geometry.hidden), want)
    }

    /// The layer norms obey the same law as the final one — the loader has
    /// one law, not one per call site.
    func testTheLayerNormsObeyTheSameLawAsTheFinalNorm() throws {
        let geometry = try QWModelGeometry.stubTinyLlamaGeometry()
        let (layers, finalNorm) = try loadStandard(geometry: geometry)
        let want = gammaVector(geometry.hidden).map { $0 - 1.0 }
        XCTAssertEqual(f16Values(layers[0].ln1, count: geometry.hidden), want)
        XCTAssertEqual(f16Values(layers[0].ln2, count: geometry.hidden), want)
        XCTAssertEqual(f16Values(layers[0].ln1, count: geometry.hidden),
                       f16Values(finalNorm, count: geometry.hidden))
    }

    /// The law follows the family: the 27B's checkpoint already stores
    /// `w = γ − 1`, so its gammas pass through untouched; a standard
    /// checkpoint stores γ, so it becomes `γ − 1`.
    func testTheGammaLawFollowsTheFamily() throws {
        let gammas = gammaVector(6)
        let planes = [Pack.finalNorm: Plane.raw(gammas)]
        let standard = try makeLoader(geometry: QWModelGeometry.stubTinyLlamaGeometry(),
                                      planes: planes)
        let standardBits = try standard.normGammaBits(Pack.finalNorm, gammas.count)
        XCTAssertEqual(standardBits, gammas.map { QWQuant.f32tof16Bits($0 - 1.0) })

        let hybrid = try makeLoader(geometry: QWModelGeometry.qwen38_27b, planes: planes)
        let hybridBits = try hybrid.normGammaBits(Pack.finalNorm, gammas.count)
        XCTAssertEqual(hybridBits, gammas.map { QWQuant.f32tof16Bits($0) })
    }

    /// The q/k norm is ONE vector for every head: upstream's
    /// `Qwen3RMSNorm(self.head_dim)` stores a [head_dim] weight, and the
    /// kernel reads it un-offset. The k norm is tiled to the kv heads.
    func testTheQKNormIsOneVectorSharedByEveryHead() throws {
        let geometry = try QWModelGeometry.stubTinyQwen3Geometry()
        var planes = standardPlanes(geometry)
        planes[Pack.qNorm] = .raw(gammaVector(geometry.headDim))
        planes[Pack.kNorm] = .raw(gammaVector(geometry.headDim))
        var L = try makeLoader(geometry: geometry, planes: planes)
        _ = try L.q4(Pack.head, n: geometry.vocab, k: geometry.hidden)
        let (layers, _) = try L.loadDecoderWeights(geometry: geometry)
        let attn = try XCTUnwrap(layers[0].attn)
        let wq = try XCTUnwrap(attn.wq)
        let wk = try XCTUnwrap(attn.wkTiled)
        XCTAssertEqual(wq.shape.elementCount, geometry.headDim)
        XCTAssertEqual(wk.shape.elementCount, geometry.kvHeads * geometry.headDim)
        // The gamma is the model's own: a standard checkpoint stores γ and
        // the engine's law is 1 + w, so the loader stores γ − 1.
        let want = gammaVector(geometry.headDim).map { $0 - 1.0 }
        XCTAssertEqual(f16Values(wq, count: geometry.headDim), want)
        // Every kv head gets the SAME vector — the k norm is shared too.
        let kAll = f16Values(wk, count: geometry.kvHeads * geometry.headDim)
        for h in 0..<geometry.kvHeads {
            let block = Array(kAll[(h * geometry.headDim)..<((h + 1) * geometry.headDim)])
            XCTAssertEqual(block, want, "kv head \(h) must hold the same k norm")
        }
    }

    /// A per-head q norm is a DIFFERENT family's layout (Olmo2's). The
    /// loader reads exactly [headDim], so a checkpoint that carries one is
    /// refused by name — never half-read.
    func testAPerHeadQNormIsRefused() throws {
        let geometry = try QWModelGeometry.stubTinyQwen3Geometry()
        var planes = standardPlanes(geometry)
        planes[Pack.qNorm] = .raw(gammaVector(geometry.qHeads * geometry.headDim))
        var L = try makeLoader(geometry: geometry, planes: planes)
        _ = try L.q4(Pack.head, n: geometry.vocab, k: geometry.hidden)
        XCTAssertThrowsError(try L.loadDecoderWeights(geometry: geometry)) { e in
            XCTAssertTrue("\(e)".contains("q_norm"), "\(e)")
        }
    }
}
