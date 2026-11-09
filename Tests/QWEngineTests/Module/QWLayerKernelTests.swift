// QWLayerKernelTests.swift — The layer-kind dispatch and the third module.
//
// These tests never touch a model pack: the third module is a toy whose
// plan is built from a descriptor, and the registry is pure.

import XCTest
import QWCore
import QWHead
@testable import QWEngine

final class QWLayerKernelTests: XCTestCase {
    /// The kinds this engine serves. Everything else is absent on purpose:
    /// a kernel that runs the wrong numerics is worse than no kernel.
    private static let served: Set<QWEnginePlan.LayerKernelKey> = [
        .init(attention: .gdn, mlp: .dense, ple: false),
        .init(attention: .gdn, mlp: .moe, ple: false),
        .init(attention: .fullAttention, mlp: .dense, ple: false),
        .init(attention: .fullAttention, mlp: .moe, ple: false),
        .init(attention: .qsaSparse, mlp: .dense, ple: false),
        .init(attention: .qsaSparse, mlp: .moe, ple: false),
    ]

    func testEveryKindIsEitherServedOrAbsent() {
        // The lists are derived from the kinds themselves, so a kind added
        // later cannot slip past: it must be served or it must be absent.
        for attention in QWEnginePlan.LayerAttentionKind.allCases {
            for mlp in QWEnginePlan.LayerMLPKind.allCases {
                for ple in [false, true] {
                    let key = QWEnginePlan.LayerKernelKey(attention: attention, mlp: mlp, ple: ple)
                    let kernel = QWLayerKernelRegistry.kernel(for: key)
                    if Self.served.contains(key) {
                        XCTAssertNotNil(kernel,
                                        "no kernel registered for \(key.attention.rawValue)/\(key.mlp.rawValue)")
                    } else {
                        XCTAssertNil(kernel,
                                     "a kernel is registered for \(key.attention.rawValue)/\(key.mlp.rawValue)")
                    }
                }
            }
        }
    }

    func testThePlanTimeResolutionServesQSAPlans() throws {
        let qsaPlan = try ToyModule.makePlan(hidden: 64, layers: 2, kvStorageMode: "f32",
                                             capabilities: [.qsaIndexer])
        XCTAssertEqual(qsaPlan.layers.first?.attention, .qsaSparse)
        let kernels = try QWLayerKernelRegistry.resolve(plan: qsaPlan)
        XCTAssertEqual(kernels.count, qsaPlan.layers.count)
    }

    func testThePlanTimeResolutionServesAPlanTheEngineCanServe() throws {
        let plan = try ToyModule.makePlan(hidden: 64, layers: 2, kvStorageMode: "f32")
        let kernels = try QWLayerKernelRegistry.resolve(plan: plan)
        XCTAssertEqual(kernels.count, plan.layers.count)
    }

    func testThirdModuleWithNoGDNLoadsAndDecodes() throws {
        // A toy Llama-style config: 4 layers, all full attention.
        let module = try ToyModule.make(hidden: 64, layers: 4, kvStorageMode: "f32")
        let state = try module.makeSessionState()
        _ = try module.encodePrefill(state, tokens: [1, 2, 3], chunk: 0..<3, runsHead: true)
        let token = try module.encodeDecodeStep(state, token: 3, position: 3)
        XCTAssertGreaterThanOrEqual(token, 0)
        XCTAssertLessThan(token, module.plan.vocab)
        XCTAssertEqual(ToyModule.supportedKVStorageModes, ["f32"],
                       "the toy module must declare a KV mode the dense model lacks")
    }

    func testTheThirdModulesPlanHasNoGDNAtAll() throws {
        let module = try ToyModule.make(hidden: 64, layers: 4, kvStorageMode: "f32")
        XCTAssertEqual(module.plan.layers.count, 4)
        for layer in module.plan.layers {
            XCTAssertEqual(layer.attention, .fullAttention)
            XCTAssertEqual(layer.mlp, .dense)
            XCTAssertFalse(layer.hasPLE)
        }
        // Its layers resolve through the production registry.
        XCTAssertEqual(module.layerKernels.count, 4)
        for kernel in module.layerKernels {
            XCTAssertEqual(type(of: kernel).key.attention, .fullAttention)
        }
    }

    func testTheThirdModulesLayersDecideItsOutput() throws {
        // The plan drives the numerics: a deeper toy answers differently.
        let shallow = try ToyModule.make(hidden: 64, layers: 2, kvStorageMode: "f32")
        let deep = try ToyModule.make(hidden: 64, layers: 6, kvStorageMode: "f32")
        let shallowState = try shallow.makeSessionState()
        let deepState = try deep.makeSessionState()
        _ = try shallow.encodePrefill(shallowState, tokens: [1, 2, 3], chunk: 0..<3, runsHead: true)
        _ = try deep.encodePrefill(deepState, tokens: [1, 2, 3], chunk: 0..<3, runsHead: true)
        let shallowToken = try shallow.encodeDecodeStep(shallowState, token: 3, position: 3)
        let deepToken = try deep.encodeDecodeStep(deepState, token: 3, position: 3)
        XCTAssertNotEqual(shallowToken, deepToken,
                          "the layer count must reach the module's output")
    }

    func testTheDenseEngineResolvesItsKernelsAtPlanTime() throws {
        // The 27B plan names GDN and full attention; both have kernels, and
        // the table is built once when the model is loaded.
        let plan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                    geometryDigest: "d", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let kernels = try QWLayerKernelRegistry.resolve(plan: plan)
        XCTAssertEqual(kernels.count, plan.layers.count)
        XCTAssertEqual(kernels.count, QWModelGeometry.qwen38_27b.numLayers)
    }

    func testTheDenseModuleRefusesARequestItCannotServe() throws {
        // The engine-less module is enough to check the contract: a sampled
        // request, a repetition penalty the greedy head cannot apply and a
        // speculative round are all refused — never quietly downgraded.
        let plan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                    geometryDigest: "d", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let module = DenseModule(plan: plan)
        XCTAssertNoThrow(try module.checkContract(sampling: .greedy,
                                                   speculative: QWSpeculativeParameters(enabled: false,
                                                                                      mtpDrafts: 0)))
        XCTAssertThrowsError(try module.checkContract(
            sampling: QWSamplerConfig(temperature: 0.8, topK: 40, topP: 0.9),
            speculative: QWSpeculativeParameters(enabled: false, mtpDrafts: 0)))
        XCTAssertThrowsError(try module.checkContract(
            sampling: QWSamplerConfig(temperature: 1, topK: 1, topP: 1, repetitionPenalty: 1.1),
            speculative: QWSpeculativeParameters(enabled: false, mtpDrafts: 0)))
        XCTAssertThrowsError(try module.checkContract(
            sampling: .greedy,
            speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 1)))
    }

    func testTheDenseModuleRefusesToConfigureACacheItWouldNotUse() throws {
        // The seam's prefill does not consult the prefix cache. Configuring
        // one would promise a cache that never answers.
        let plan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                    geometryDigest: "d", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let module = DenseModule(plan: plan)
        XCTAssertFalse(module.hasPrefixCache)
        XCTAssertThrowsError(try module.configurePrefixCache(QWCacheConfig())) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
    }

    func testTheCoreCallsFinishWhenTheRequestIsOver() throws {
        // The module must be able to give the engine back when the request
        // ends — that is where the dense module releases its permit.
        let plan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                    geometryDigest: "d", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let module = LoopToyModule(plan: plan)
        let core = try QWEngineCore(configuration: .init(plan: plan, module: module,
                                                          sampling: .greedy, maxSessions: 1))
        _ = try core.generate(prompt: [1, 2, 3], maxNew: 4, stopIDs: [],
                              sampling: .greedy, onToken: nil)
        XCTAssertEqual(module.finishedRequests, 1)
    }

    func testCompiledLayerStackUnifies4SeamsAndRollbackRing() throws {
        let densePlan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy-27b",
                                         geometryDigest: "d", prefillChunkRows: 512,
                                         kvStorageMode: "f16")
        let denseStack = DenseModule(plan: densePlan).compiledStack
        XCTAssertEqual(denseStack.layers.count, QWModelGeometry.qwen38_27b.numLayers)
        XCTAssertTrue(denseStack.layers.allSatisfy { $0.streamCarrier == .singleStream && $0.channelMixer == .dense && !$0.hasPLE })
        XCTAssertEqual(denseStack.cuts, [0, denseStack.layers.count])
        XCTAssertEqual(denseStack.steps, [[.commit]])
        XCTAssertTrue(QWCompiledLayerStack.supportsRollbackWindow(4))
        XCTAssertTrue(QWCompiledLayerStack.supportsRollbackWindow(8))
        XCTAssertFalse(QWCompiledLayerStack.supportsRollbackWindow(9))
    }
}
