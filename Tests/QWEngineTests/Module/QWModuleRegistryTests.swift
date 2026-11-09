import XCTest
import QWCore
@testable import QWEngine

/// The registry is the one admission point: a family names its module, and
/// a module declares what it can actually do.
final class QWModuleRegistryTests: XCTestCase {
    /// A dense 27B-shaped plan, built from the compiled-in geometry.
    private func densePlan(kvStorageMode: String = "f16") throws -> QWEnginePlan {
        try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "qwen3.8-27b",
                         geometryDigest: "d", prefillChunkRows: 512,
                         kvStorageMode: kvStorageMode)
    }

    /// A Flash-Next-shaped plan. The descriptor is built in memory, so the
    /// suite never reads an installed pack.
    private func flashNextDescriptor(advertising: Set<QWModelDescriptor.Capability>) -> QWModelDescriptor {
        QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: advertising,
            hidden: 2560, vocab: 248320,
            denseIntermediate: nil, numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4, rmsEps: 1e-6,
            maxPositionEmbeddings: 262144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 512, expertsPerTok: 10, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320,
            gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10240, gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248044)
    }

    private func flashNextPlan(advertising: Set<QWModelDescriptor.Capability>) throws -> QWEnginePlan {
        let descriptor = flashNextDescriptor(advertising: advertising)
        return try QWEnginePlan(geometry: QWModelGeometry(descriptor: descriptor),
                                modelId: "qwen3.8-flash-next", geometryDigest: "d",
                                prefillChunkRows: 512, kvStorageMode: "f16")
    }

    func testEverySupportedFamilyResolvesToAModule() throws {
        for family in QWModuleRegistry.supportedFamilies {
            XCTAssertNotNil(QWModuleRegistry.moduleType(for: family), family.rawValue)
        }

        XCTAssertEqual(QWModuleRegistry.supportedFamilies.count, 7)
        XCTAssertEqual(Set(QWModuleRegistry.supportedFamilies),
                       [.qwen38Dense, .qwen38FlashNext, .qwen2Dense, .llamaDense,
                        .mistralDense, .qwen3Dense, .qwen3MoE])
    }

    func testSpeculationPolicyForInitialLoadAndModelSwitch() throws {
        let flash = flashNextDescriptor(advertising: [.mtp])
        let requested = QWSpeculativeParameters(mtpDrafts: 7, mtpBackend: "dflash2")
        let switched = try FlashNextModule.resolveSpeculation(
            requested, descriptor: flash, environment: ["FN_DRAFT": "ngram"], modelSwitch: true)
        XCTAssertEqual(switched.parameters.mtpBackend, "variational")
        XCTAssertEqual(switched.parameters.mtpDrafts, 3)
        XCTAssertEqual(switched.reportedDrafts, 3)
        XCTAssertEqual(switched.reportedBackend, "ngram")
        let loaded = try FlashNextModule.resolveSpeculation(
            switched.parameters, descriptor: flash, environment: ["FN_DRAFT": "ngram"], modelSwitch: false)
        XCTAssertEqual(loaded.reportedBackend, switched.reportedBackend)
        XCTAssertEqual(loaded.reportedDrafts, switched.reportedDrafts)

        let dense = try DenseModule.resolveSpeculation(
            requested, descriptor: .qwen38_27b, environment: [:], modelSwitch: false)
        XCTAssertEqual(dense.engineBackend, .dflash2)
        XCTAssertEqual(dense.engineDrafts, 7)
        let disabled = try DenseModule.resolveSpeculation(
            QWSpeculativeParameters(enabled: false, mtpDrafts: 3), descriptor: .qwen38_27b,
            environment: [:], modelSwitch: false)
        XCTAssertEqual(disabled.engineDrafts, 0)
    }

    func testSupportedFamiliesCoversEveryKnownFamily() {
        // A family with no module is not "supported". The Qwen3 families
        // landed theirs (Task 4.5a), so none is left unserved — and the
        // registry says so.
        let served = Set(QWModuleRegistry.supportedFamilies)
        let unserved = Set(QWModelDescriptor.Family.allCases).subtracting(served)
        XCTAssertEqual(served, [.qwen38Dense, .qwen38FlashNext,
                                 .qwen2Dense, .llamaDense, .mistralDense,
                                 .qwen3Dense, .qwen3MoE])
        XCTAssertEqual(unserved, [])
        for family in unserved {
            XCTAssertNil(QWModuleRegistry.moduleType(for: family), family.rawValue)
        }
    }

    func testTheQwen3FamiliesDeclareTheStandardContract() {
        // The standard families ship none of the 27B's hybrid-only extras (DFlash, MTP, vision, GDN prefill);
        // they support universal Q8 and Q4 KV cache quantization and model-agnostic LSA.
        let expected: Set<QWKnobCapability> = [.kvQuantQ8, .kvQuantQ4, .lsa]
        XCTAssertEqual(QWModuleRegistry.capabilities(for: .qwen3Dense), expected)
        XCTAssertEqual(QWModuleRegistry.capabilities(for: .qwen3MoE), expected)
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3Dense))
        XCTAssertNotNil(QWModuleRegistry.moduleType(for: .qwen3MoE))
    }

    func testUnknownPackModelIdYieldsNilNotATypedError() {
        XCTAssertNil(QWModuleRegistry.moduleType(forPackModelId: "not-a-model"))
        XCTAssertNil(QWModuleRegistry.family(forPackModelId: "not-a-model"))
    }

    func testPackModelIdsMapToTheirFamily() {
        XCTAssertEqual(QWModuleRegistry.family(forPackModelId: "qwen3.8-27b"), .qwen38Dense)
        XCTAssertEqual(QWModuleRegistry.family(forPackModelId: "qwen3.8-flash-next"), .qwen38FlashNext)
        XCTAssertNotNil(QWModuleRegistry.moduleType(forPackModelId: "qwen3.8-flash-next"))
    }

    func testDenseModuleDeclaresItsKVStorageModes() throws {
        let type = try XCTUnwrap(QWModuleRegistry.moduleType(for: .qwen38Dense))
        XCTAssertTrue(type.supportedKVStorageModes.contains("f16"))
        XCTAssertTrue(type.supportedKVStorageModes.contains("q8"))
        XCTAssertFalse(type.supportedKVStorageModes.contains("f32"),
                       "the dense engine has no f32 KV path")
    }

    func testFlashNextModuleDeclaresWiderKVStorageModes() throws {
        let type = try XCTUnwrap(QWModuleRegistry.moduleType(for: .qwen38FlashNext))
        XCTAssertTrue(type.supportedKVStorageModes.contains("f32"))
        XCTAssertTrue(type.supportedKVStorageModes.contains("q4"))
    }

    func testCapabilitiesSayWhatEachEngineCanDo() {
        let dense = DenseModule.capabilities
        let flash = FlashNextModule.capabilities
        XCTAssertTrue(dense.contains(.dflash), "the DFlash drafter is dense-only")
        XCTAssertFalse(flash.contains(.dflash))
        XCTAssertTrue(dense.contains(.ropeYaRN))
        XCTAssertFalse(flash.contains(.ropeYaRN), "Flash-Next has no rope-YaRN path")
        XCTAssertFalse(flash.contains(.gdnPrefill),
                       "Flash-Next echoes gdnPrefill, it does not choose it")
        XCTAssertTrue(dense.contains(.gdnPrefill))
    }

    func testFlashNextAndDenseSupportVision() {
        XCTAssertTrue(FlashNextModule.capabilities.contains(.vision),
                      "Flash-Next (qwen4_exp) supports the 27-block ViT vision tower")
        XCTAssertTrue(DenseModule.capabilities.contains(.vision),
                      "the dense pack advertises a vision tower")
    }

    func testAdaptiveMTPIsDenseOnly() {
        // Flash-Next speculates but has no adaptive-depth or entropy gate.
        XCTAssertTrue(DenseModule.capabilities.contains(.mtpAdaptive))
        XCTAssertFalse(FlashNextModule.capabilities.contains(.mtpAdaptive))
        XCTAssertTrue(FlashNextModule.capabilities.contains(.speculative))
    }

    func testEveryFamilyDeclaresACapabilitySetAndADoorSet() throws {
        for family in QWModuleRegistry.supportedFamilies {
            let type = try XCTUnwrap(QWModuleRegistry.moduleType(for: family))
            // The standard families declare no knobs: an empty capability set
            // is their declaration, not a missing one.
            if !(type is StandardModule.Type) {
                XCTAssertFalse(type.capabilities.isEmpty, family.rawValue)
            }
            XCTAssertFalse(type.requiredDoors.isEmpty, family.rawValue)
            XCTAssertFalse(type.supportedKVStorageModes.isEmpty, family.rawValue)
        }
    }

    func testDoorsAreDistinctWithinAModule() {
        for type in [DenseModule.self as any QWInferenceModule.Type,
                     FlashNextModule.self as any QWInferenceModule.Type,
                     Qwen3DenseModule.self as any QWInferenceModule.Type,
                     Qwen3MoEModule.self as any QWInferenceModule.Type] {
            XCTAssertEqual(Set(type.requiredDoors).count, type.requiredDoors.count,
                           "a repeated door would prewarm twice")
        }
    }

    func testRegistryBuildsTheModuleThatServesThePlan() throws {
        let plan = try densePlan()
        let module = try QWModuleRegistry.makeModule(plan: plan, engine: nil)
        XCTAssertEqual(module.plan.modelId, "qwen3.8-27b")
        XCTAssertTrue(module is DenseModule)
        XCTAssertEqual(module.kvStorageMode, "f16")
    }

    func testRegistryBuildsTheFlashNextModuleForItsOwnFamily() throws {
        let plan = try flashNextPlan(advertising: [.mtp])
        let module = try QWModuleRegistry.makeModule(plan: plan, engine: nil)
        XCTAssertTrue(module is FlashNextModule)
    }

    func testAModuleThatCannotHonourAnAdvertisedCapabilityIsRefused() throws {
        // Flash-Next supports vision, so advertising [.vision, .mtp] succeeds.
        let fnVisionPlan = try flashNextPlan(advertising: [.vision, .mtp])
        XCTAssertTrue(try QWModuleRegistry.makeModule(plan: fnVisionPlan, engine: nil) is FlashNextModule)

        // Standard qwen3-dense has no vision tower; advertising .vision is refused.
        let stdDesc = QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3",
            capabilities: [.vision],
            hidden: 2560, vocab: 151936,
            denseIntermediate: 9728, numLayers: 4,
            layerTypes: ["full_attention", "full_attention", "full_attention", "full_attention"],
            fullAttentionInterval: 1, rmsEps: 1e-6,
            maxPositionEmbeddings: 32768,
            qHeads: 32, kvHeads: 8, headDim: 128,
            attnOutputGate: false, partialRotaryFactor: 1.0, ropeTheta: 1e6,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0,
            gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 151645
        )
        let stdVisionPlan = try QWEnginePlan(
            geometry: QWModelGeometry(descriptor: stdDesc),
            modelId: "qwen3-dense",
            geometryDigest: "std-digest",
            prefillChunkRows: 512,
            kvStorageMode: "f16"
        )
        XCTAssertThrowsError(try QWModuleRegistry.makeModule(plan: stdVisionPlan, engine: nil)) { error in
            XCTAssertEqual(error as? QWModuleError,
                           .unsupportedCapability(model: "qwen3-dense", capability: .vision))
        }
    }

    func testAdvertisedCapabilitiesMapToTheKnobCapabilitiesTheyNeed() {
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.vision]), [.vision])
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.mtp]), [.speculative])
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.moe, .ple]), [])
    }

    func testModuleStubsFailLoudRatherThanPretendingToWork() throws {
        let plan = try densePlan()
        let module = DenseModule(plan: plan)
        XCTAssertThrowsError(try module.makeSessionState()) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
        XCTAssertThrowsError(try module.configurePrefixCache(QWCacheConfig())) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
    }

    func testTheKVModeTravelsWithThePlan() throws {
        let plan = try densePlan(kvStorageMode: "q4")
        XCTAssertEqual(plan.kvStorageMode, "q4")
        let module = try QWModuleRegistry.makeModule(plan: plan, engine: nil)
        XCTAssertEqual(module.kvStorageMode, "q4")
        XCTAssertTrue(DenseModule.supportedKVStorageModes.contains(module.kvStorageMode))
    }

    func testUnifiedQWEngineHandleAndBatchScheduler() throws {
        let desc = flashNextDescriptor(advertising: [.mtp, .hyperConnections, .qsaIndexer, .moe])
        let plan = try QWEnginePlan(geometry: QWModelGeometry(descriptor: desc),
                                    modelId: "qwen3.8-flash-next", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        let flashModule = FlashNextModule(plan: plan)
        let unified = QWEngine(engine: flashModule, descriptor: desc, plan: plan)
        XCTAssertEqual(unified.plan.modelId, "qwen3.8-flash-next")
        XCTAssertEqual(unified.compiledStack.layers.count, 4)
        XCTAssertTrue(unified.compiledStack.layers.allSatisfy {
            $0.streamCarrier == .hyperConnection && $0.channelMixer == .moe
        })
        XCTAssertEqual(unified.compiledStack.layers[0].sequenceMixer, .gdn)
        XCTAssertEqual(unified.compiledStack.layers[3].sequenceMixer, .qsaSparse)
        XCTAssertEqual(unified.batchScheduler.inFlightRequests, 0)
        XCTAssertTrue(unified.supportsTokenConstraints)
    }
}
