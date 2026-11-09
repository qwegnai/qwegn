// QWStandardGeometryLawTests.swift — the standard path's geometry laws:
// the scratch, the prefill pad and the dispatch grids must be computed
// from the model's own numbers, never the 27B's.
import XCTest
import Metal
@testable import QWEngine
import QWCore
import QWHAL
@testable import QWAttn

final class QWStandardGeometryLawTests: XCTestCase {

    /// The geometries a standard checkpoint actually ships: llama-3.1-8B,
    /// qwen2-7B, qwen3-1.7B and qwen3-30B-a3b, plus the 27B.
    private struct Shape {
        let name: String
        let hidden: Int
        let intermediate: Int
        let qHeads: Int
        let kvHeads: Int
        let headDim: Int
        let qGate: Bool
    }

    private static let shapes: [Shape] = [
        .init(name: "llama-3.1-8b", hidden: 4096, intermediate: 14336,
              qHeads: 32, kvHeads: 8, headDim: 128, qGate: false),
        .init(name: "qwen2-7b", hidden: 3584, intermediate: 18944,
              qHeads: 28, kvHeads: 4, headDim: 128, qGate: false),
        .init(name: "qwen3-1.7b", hidden: 2048, intermediate: 9728,
              qHeads: 16, kvHeads: 8, headDim: 128, qGate: false),
        .init(name: "qwen3-30b-a3b", hidden: 2048, intermediate: 6144,
              qHeads: 32, kvHeads: 4, headDim: 128, qGate: false),
    ]

    /// The 27B's numbers, spelled out so the laws below can be checked
    /// against them rather than against the code that reads them.
    private static let b27 = QWModelGeometry.qwen38_27b

    /// The decode partials scratch must be sized from the model's own heads
    /// and head width — it used to be sized from the 27B's default, so a
    /// model with more than 24 heads of 256 halves could not be served.
    func testThePartialsScratchIsSizedForTheModelsOwnGeometry() {
        for s in Self.shapes {
            let g = Self.geometry(s)
            let bytes = QWAttnDecode.decodePartialsBytes(geometry: g)
            let perHead = QWAttnDecode.partialRowFloats(geometry: g) * 4
            XCTAssertEqual(bytes, g.qHeads * QWAttnDecode.maxSplits * perHead,
                "\(s.name): the partials scratch must be qHeads × splits × the head's own row")
            XCTAssertNotEqual(bytes, QWAttnDecode.decodePartialsBytes(geometry: Self.b27),
                "\(s.name): the partials scratch must not be the 27B's size")
        }
    }

    /// The decode scratch's budget must be the model's own — the driver's
    /// region size and its two allocations all come from this one law.
    func testTheDecodeScratchBudgetIsTheModelsOwn() {
        for s in Self.shapes {
            let g = Self.geometry(s)
            let budget = QWEngineDriver.partialsScratchBytes(geometry: g, mtpDrafts: 0)
            XCTAssertEqual(budget, QWAttnDecode.decodePartialsBytes(geometry: g))
            XCTAssertNotEqual(budget, QWEngineDriver.partialsScratchBytes(geometry: Self.b27, mtpDrafts: 0),
                "\(s.name): the decode scratch must not be the 27B's size")
            XCTAssertEqual(QWEngineDriver.partialsScratchBytes(geometry: g, mtpDrafts: 3), 2 * budget)
        }
    }

    /// Metal accepts a `dispatchThreads` grid that is not a multiple of
    /// the threadgroup and **rounds it up** — the two forms agree byte
    /// for byte. The engine does not lean on that leniency.
    func testTheGridRoundingIsTheSameDispatch() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let pso = try hal.pipeline("qwg_cast_f32_to_f16")
        let dev = hal.device
        let words = 256
        let a = dev.makeBuffer(length: words * 4, options: .storageModeShared)!
        let b = dev.makeBuffer(length: words * 4, options: .storageModeShared)!
        let src = dev.makeBuffer(length: words * 4, options: .storageModeShared)!
        let s = src.contents().bindMemory(to: Float.self, capacity: words)
        for i in 0..<words { s[i] = Float(i + 1) }
        var nF8: UInt32 = 3
        func run(_ dst: MTLBuffer, groups: Bool) {
            guard let (cb, enc) = hal.begin() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(src, offset: 0, index: 0)
            enc.setBuffer(dst, offset: 0, index: 1)
            enc.setBytes(&nF8, length: 4, index: 2)
            if groups {
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else {
                enc.dispatchThreads(MTLSize(width: 3, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            hal.end(cb, enc)
            cb.waitUntilCompleted()
        }
        run(a, groups: false)
        run(b, groups: true)
        let pa = a.contents().bindMemory(to: UInt32.self, capacity: words / 4)
        let pb = b.contents().bindMemory(to: UInt32.self, capacity: words / 4)
        var differing = 0
        for i in 0..<(words / 4) where pa[i] != pb[i] { differing += 1 }
        XCTAssertEqual(differing, 0,
            "a rounded grid and a ragged dispatchThreads grid must agree byte for byte")
        // The comparison above is vacuous if BOTH buffers stayed zero, so
        // the conversion itself must be proven to have happened.
        let halves = b.contents().bindMemory(to: UInt16.self, capacity: words / 2)
        var nonzero = 0
        for i in 0..<(words / 2) where halves[i] != 0 { nonzero += 1 }
        XCTAssertGreaterThan(nonzero, 0, "the cast must have written something")
    }

    /// The 27B's own numbers must be unchanged by the law — PINNED to a
    /// literal, not to a tautology. 24 heads x 64 splits x 258 floats x 4 B.
    func testThe27BPartialsScratchIsUnchanged() {
        XCTAssertEqual(QWAttnDecode.decodePartialsBytes(geometry: Self.b27), 1_585_152)
        XCTAssertEqual(QWAttnDecode.decodePartialsBytesBatched(T: 1, geometry: Self.b27), 1_585_152)
        XCTAssertEqual(QWAttnDecode.decodePartialsBytesBatched(T: 2, geometry: Self.b27), 3_170_304)
    }

    /// The budget and the carve are ONE expression: the region size and
    /// every carve read the same number, so they cannot drift apart.
    func testTheBudgetAndTheCarveAreOneNumber() {
        for s in Self.shapes {
            let g = Self.geometry(s)
            let carve = QWEngineDriver.partialsCarveBytes(geometry: g)
            XCTAssertEqual(carve, QWAttnDecode.decodePartialsBytes(geometry: g),
                "\(s.name): the carve is the decode's own partials size")
            XCTAssertEqual(QWEngineDriver.partialsScratchBytes(geometry: g, mtpDrafts: 0), carve)
            XCTAssertEqual(QWEngineDriver.partialsScratchBytes(geometry: g, mtpDrafts: 1), 2 * carve)
            XCTAssertEqual(QWEngineDriver.partialsScratchBytes(geometry: g, mtpDrafts: 3), 2 * carve,
                "the draft region is ONE more, however many drafts run")
        }
    }

    /// The group-sum scratch's widths come from the model's own GEMV
    /// shapes — the o_proj's width is the larger of the attention's and
    /// the GDN's, and for a standard family the GDN's is ZERO, not 48.
    func testTheGroupSumWidthsAreTheModelsOwn() {
        for s in Self.shapes {
            let g = Self.geometry(s)
            let oProj = QWEngineDriver.oProjGroupSumFloats(geometry: g)
            XCTAssertGreaterThanOrEqual(oProj, (g.hidden + 127) / 128,
                "\(s.name): the o_proj's group sums must cover the attention's o_proj")
            XCTAssertNotEqual(oProj, 0, "\(s.name): the o_proj's group sums must not vanish")
        }
        // The 27B's own widths are unchanged.
        XCTAssertEqual(QWEngineDriver.oProjGroupSumFloats(geometry: Self.b27), 48)
        XCTAssertEqual(QWEngineDriver.groupSumFloats(k: 5120), 64)
        XCTAssertEqual(QWEngineDriver.groupSumFloats(k: 6144), 64)
        XCTAssertEqual(QWEngineDriver.groupSumFloats(k: 17408), 192)
    }

    /// The prefill scratch's per-token footprint is derived from the
    /// model's own geometry — the 27B's 760_000-byte estimate tracked no
    /// model at all.
    func testThePrefillFootprintIsDerivedFromTheGeometry() {
        for s in Self.shapes {
            let g = Self.geometry(s)
            let bytes = QWEngineDriver.prefillBytesPerToken(geometry: g)
            XCTAssertGreaterThan(bytes, 0, "\(s.name): the footprint must be positive")
            XCTAssertNotEqual(bytes, QWEngineDriver.prefillBytesPerToken(geometry: Self.b27),
                "\(s.name): the footprint must not be the 27B's")
        }
        // The 27B's own: 161_888 f16 + 106_080 f32 elements.
        XCTAssertEqual(QWEngineDriver.prefillBytesPerToken(geometry: Self.b27), 748_096)
    }

    /// The 27B's prefill pad writes with the 27B's row strides, so it
    /// serves that geometry only. Every standard geometry must be refused
    /// by it — running it would pad at the wrong offsets, past the row.
    func testTheZeroPadServesThe27BStridesOnly() {
        XCTAssertTrue(QWEngineDriver.zeroPadServes(geometry: Self.b27),
            "the 27B's own strides must be served")
        for s in Self.shapes {
            XCTAssertFalse(QWEngineDriver.zeroPadServes(geometry: Self.geometry(s)),
                "\(s.name): the 27B's pad must not serve this geometry")
        }
    }

    /// The SwiGLU grid must cover the intermediate. The audit's "it aborted
    /// the dispatch" claim was WRONG: Metal rounds a ragged grid up.
    func testTheSiluGridCoversAnyIntermediate() {
        for i in [1024, 11008, 18944, 9728, 6144, 14336, 17408, 96, 8, 0] {
            let tgs = QWEngineDriver.siluThreadgroups(i: i)
            let threads = tgs * 128
            let needed = (i + 7) / 8
            XCTAssertGreaterThanOrEqual(threads, needed, "I=\(i): the grid must cover the data")
            XCTAssertLessThan(threads - needed, 128, "I=\(i): the tail must stay under one threadgroup")
        }
    }

    /// The 27B's intermediate must land on the same grid it always did.
    func testTheSiluGridIsUnchangedForThe27B() {
        XCTAssertEqual(QWEngineDriver.siluThreadgroups(i: 17408), 17)
    }

    /// The pad's grid must be a whole number of threadgroups that still
    /// covers the q width.
    func testTheZeroPadGridCoversAnyQWidth() {
        for q in [12288, 4096, 3584, 896, 6144, 128, 64] {
            let tgs = QWEngineDriver.zeroPadThreadgroups(qProjOut: q)
            XCTAssertGreaterThanOrEqual(tgs * 256, q, "q=\(q): the grid must cover the q width")
            XCTAssertLessThan(tgs * 256 - q, 256, "q=\(q): the tail must stay under one threadgroup")
        }
        XCTAssertEqual(QWEngineDriver.zeroPadThreadgroups(qProjOut: 12288), 48)
    }

    /// A geometry built from a shape, through the same descriptor the
    /// standard decoder produces.
    private static func geometry(_ s: Shape) -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": s.hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": s.qHeads,
            "num_key_value_heads": s.kvHeads,
            "head_dim": s.headDim,
            "intermediate_size": s.intermediate,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try! JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try! QWModelDescriptor.decode(configJSON: data))
    }
}
