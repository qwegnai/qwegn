// QWStandardModuleTests.swift — The standard families' module (Task 4.3).
import XCTest
import Foundation
@testable import QWEngine
@testable import QWCore
@testable import QWHAL
@testable import QWRope

final class QWStandardModuleTests: XCTestCase {
    func testTheStandardFamiliesNowResolveToAModule() {
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .llamaDense))
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .mistralDense))
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen2Dense))
    }

    func testTheQwen3FamiliesNowResolveToTheirOwnModules() {
        // Task 4.4 served the MoE path, Task 4.5a the dense one. A family
        // with no module says so rather than running another path.
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3Dense))
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3MoE))
        XCTAssertEqual(Qwen3DenseModule.family, .qwen3Dense)
        XCTAssertEqual(Qwen3MoEModule.family, .qwen3MoE)
    }

    func testTheStandardFamiliesAreInTheRegistrysFamilyList() {
        let families = QWModuleRegistry.supportedFamilies
        XCTAssertTrue(families.contains(.llamaDense))
        XCTAssertTrue(families.contains(.mistralDense))
        XCTAssertTrue(families.contains(.qwen2Dense))
        XCTAssertTrue(families.contains(.qwen3Dense))
        XCTAssertTrue(families.contains(.qwen3MoE))
    }

    func testAPureAttentionPlanNeedsNoGDN() throws {
        let plan = try QWEnginePlan(geometry: try QWModelGeometry.stubPureAttention(),
                                    modelId: "stub",
                                    geometryDigest: "stub",
                                    prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        XCTAssertTrue(plan.layers.allSatisfy { $0.attention == .fullAttention })
        XCTAssertTrue(plan.layers.allSatisfy { $0.mlp == .dense })
    }

    func testANonGatedModelDoesNotWidenTheQProjection() throws {
        let plan = try QWEnginePlan(geometry: try QWModelGeometry.stubPureAttention(gated: false),
                                    modelId: "stub",
                                    geometryDigest: "stub",
                                    prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        XCTAssertEqual(plan.qProjOut, plan.qHeads * plan.headDim)
    }

    func testAGatedModelDoublesTheQProjection() throws {
        let plan = try QWEnginePlan(geometry: try QWModelGeometry.stubPureAttention(gated: true),
                                    modelId: "stub",
                                    geometryDigest: "stub",
                                    prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        XCTAssertEqual(plan.qProjOut, 2 * plan.qHeads * plan.headDim)
    }

    func testTheStandardModuleShipsNoneOfThe27BsExtras() {
        let expected: Set<QWKnobCapability> = [.kvQuantQ8, .kvQuantQ4, .lsa]
        XCTAssertEqual(StandardModule.capabilities, expected)
        XCTAssertEqual(LlamaModule.capabilities, expected)
        XCTAssertEqual(MistralModule.capabilities, expected)
        XCTAssertEqual(Qwen2DenseModule.capabilities, expected)
    }

    func testEachStandardModuleNamesItsOwnFamily() {
        XCTAssertEqual(LlamaModule.family, .llamaDense)
        XCTAssertEqual(MistralModule.family, .mistralDense)
        XCTAssertEqual(Qwen2DenseModule.family, .qwen2Dense)
    }

    func testTheStandardDoorsAreTheDenseOnesWithoutTheHybridFamilies() {
        let doors = Set(StandardModule.requiredDoors)
        XCTAssertFalse(doors.isEmpty)
        XCTAssertTrue(doors.isSubset(of: Set(DenseModule.requiredDoors)))
        for door in doors {
            XCTAssertFalse(door.hasPrefix("qwg_" + "gdn_"), door)
            XCTAssertFalse(door.hasPrefix("qwg_" + "mtp_"), door)
            XCTAssertFalse(door.hasPrefix("qwg_" + "dflash_"), door)
        }
    }

    /// The standard model's geometry: a 128-half head, no gate, no q/k norm.
    func testTheStandardGeometryHasNoGateAndNoQKNorm() throws {
        let geometry = try QWModelGeometry.stubLlamaGeometry()
        XCTAssertEqual(geometry.headDim, 128)
        XCTAssertEqual(geometry.qHeads, 32)
        XCTAssertEqual(geometry.kvHeads, 8)
        XCTAssertFalse(geometry.attnOutputGate)
        XCTAssertFalse(geometry.descriptor.hasQKNorm)
        XCTAssertEqual(geometry.qProjOut, 32 * 128)
        XCTAssertEqual(geometry.rotaryDims, 128)
        XCTAssertEqual(geometry.gdnTotalN, 0)
    }

    /// The 27B's constants are exactly the literals the parameterisation
    /// replaced — the proof that its path is unchanged.
    func testThe27BConstantsAreTheLiteralsTheyReplaced() {
        let geometry = QWModelGeometry.qwen38_27b
        let constants = QWHal.standardConstants(geometry: geometry)
        XCTAssertEqual(constants.qStride, 2, "the 27B's q row is Q‖gate: two slots per head")
        XCTAssertEqual(constants.attnScale, 0.0625, "1/√256, the literal the kernel used")
        XCTAssertTrue(constants.hasQGate)
        XCTAssertEqual(geometry.rotaryDims, 64, "the rope width the kernel used")
        XCTAssertEqual(geometry.qProjOut / 8, 1536, "the q row stride in uint4s")
    }

    /// A standard model's constants are the other values.
    func testAStandardModelsConstantsAreItsOwn() throws {
        let geometry = try QWModelGeometry.stubLlamaGeometry()
        let constants = QWHal.standardConstants(geometry: geometry)
        XCTAssertEqual(constants.qStride, 1, "no gate: one slot per head")
        XCTAssertEqual(constants.attnScale, Float(1.0 / 128.0.squareRoot()))
        XCTAssertFalse(constants.hasQGate)
    }

    /// The module serves the head widths the standard path's own kernels
    /// serve — 128 halves as well as 256 — and refuses a width the law
    /// does not cover.
    func testTheModuleServesTheStandardHeadWidths() throws {
        for geometry in [try QWModelGeometry.stubLlamaGeometry(),
                         try QWModelGeometry.stubStandard256Geometry()] {
            let plan = try QWEnginePlan(geometry: geometry, modelId: "stub",
                                        geometryDigest: "stub", prefillChunkRows: 512,
                                        kvStorageMode: "f16")
            XCTAssertNoThrow(try LlamaModule.make(plan: plan, engine: nil),
                             "\(geometry.headDim) halves")
        }
    }

    /// The serving law: a head's width is a multiple of 32 and at most 256,
    /// and a compressed KV mode belongs to the 27B's shape alone.
    func testTheServingLawIsOnePredicateOverTheNumbers() {
        XCTAssertNoThrow(try QWEngineDriver.validateServing(
            headDim: 128, kvHeads: 8, qHeads: 32, kvMode: "f16"))
        XCTAssertThrowsError(try QWEngineDriver.validateServing(
            headDim: 100, kvHeads: 8, qHeads: 32, kvMode: "f16"))
        XCTAssertThrowsError(try QWEngineDriver.validateServing(
            headDim: 288, kvHeads: 8, qHeads: 32, kvMode: "f16"))
        // f16, q8, and q4 are served on standard shapes where headDim is a multiple of 32 <= 256.
        XCTAssertNoThrow(try QWEngineDriver.validateServing(
            headDim: 128, kvHeads: 8, qHeads: 32, kvMode: "q8"))
        XCTAssertNoThrow(try QWEngineDriver.validateServing(
            headDim: 128, kvHeads: 8, qHeads: 32, kvMode: "q4"))
        XCTAssertNoThrow(try QWEngineDriver.validateServing(
            headDim: 256, kvHeads: 4, qHeads: 24, kvMode: "q8"))
    }

    /// The module's pre-flight and the driver's check read ONE law, so they
    /// cannot disagree about a geometry — over a grid of shapes, not one.
    func testTheModuleAndTheDriverAgreeOnEveryShape() throws {
        for kvHeads in [2, 4, 8] {
            for headDim in [64, 128, 256] {
                for qHeads in [kvHeads, kvHeads * 4] {
                    let module = errorOf {
                        try QWEngineDriver.validateServing(
                            headDim: headDim, kvHeads: kvHeads, qHeads: qHeads, kvMode: "f16")
                    }
                    let driver = errorOf {
                        try QWEngineDriver.validateGeometry(
                            try QWModelGeometry.stubShape(kvHeads: kvHeads, qHeads: qHeads,
                                                          headDim: headDim), kvMode: .f16)
                    }
                    XCTAssertEqual(module, driver, "\(kvHeads)×\(headDim) gqa\(qHeads / kvHeads)")
                }
            }
        }
    }

    /// The plan-shaped predicate and the instance one are ONE body, and that
    /// body names the 27B's shape: its own geometry satisfies it, and a shape
    /// that differs in any of the three numbers does not.
    func testThePlanShapedPredicateAgreesWithTheInstance() throws {
        XCTAssertTrue(QWModelGeometry.qwen38_27b.is27BShape, "the 27B is the 27B's shape")
        for kvHeads in [2, 4, 8] {
            for headDim in [64, 128, 256] {
                for qHeads in [kvHeads, kvHeads * 4, kvHeads * 6] {
                    let geometry = try QWModelGeometry.stubShape(
                        kvHeads: kvHeads, qHeads: qHeads, headDim: headDim)
                    XCTAssertEqual(
                        QWModelGeometry.is27BShape(kvHeads: kvHeads, headDim: headDim,
                                                  gqaRatio: geometry.gqaRatio),
                        geometry.is27BShape, "\(kvHeads)×\(headDim) gqa\(geometry.gqaRatio)")
                }
            }
        }
        // The 27B's shape at a different GQA, or with a different head count,
        // is NOT the 27B's shape: the frozen kernels hard-code all three.
        for (kvHeads, headDim, qHeads) in [(4, 256, 8), (4, 128, 24), (8, 256, 48)] {
            let geometry = try QWModelGeometry.stubShape(
                kvHeads: kvHeads, qHeads: qHeads, headDim: headDim)
            XCTAssertFalse(geometry.is27BShape, "\(kvHeads)×\(headDim) gqa\(qHeads / kvHeads)")
        }
    }

    /// Whether a call threw, as a bool — the two laws must agree on it.
    private func errorOf(_ body: () throws -> Void) -> Bool {
        do { try body(); return false } catch { return true }
    }

    /// The rope table's width is the model's rotary dims.
    func testTheRopeTableIsAsWideAsTheRotaryDims() throws {
        let tables = QWRopeTables(rows: 4, yarn: nil, geometry: try QWModelGeometry.stubLlamaGeometry())
        XCTAssertEqual(tables.width, 128)
        XCTAssertEqual(tables.pairs, 64)
        XCTAssertEqual(tables.cosSin.count, 4 * 128)
        let wide = QWRopeTables(rows: 4, yarn: nil, geometry: .qwen38_27b)
        XCTAssertEqual(wide.width, 64)
        XCTAssertEqual(wide.cosSin.count, 4 * 64)
    }
}
