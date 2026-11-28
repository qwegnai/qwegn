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
            let mod = try XCTUnwrap(QWModuleRegistry.moduleType(for: family), family.rawValue)
            XCTAssertEqual(ObjectIdentifier(mod), ObjectIdentifier(DenseModule.self), family.rawValue)
        }

        XCTAssertEqual(QWModuleRegistry.supportedFamilies.count, 19)
        XCTAssertEqual(Set(QWModuleRegistry.supportedFamilies), Set(QWModelDescriptor.Family.allCases))
    }

    func testSpeculationPolicyForInitialLoadAndModelSwitch() throws {
        let flash = flashNextDescriptor(advertising: [.mtp])
        let requested = QWSpeculativeParameters(mtpDrafts: 7, mtpBackend: "dflash2")
        let switched = try DenseModule.resolveSpeculation(
            requested, descriptor: flash, environment: ["FN_DRAFT": "ngram"], modelSwitch: true)
        XCTAssertEqual(switched.parameters.mtpBackend, "variational")
        XCTAssertEqual(switched.parameters.mtpDrafts, 3)
        XCTAssertEqual(switched.reportedDrafts, 3)
        XCTAssertEqual(switched.reportedBackend, "ngram")
        let loaded = try DenseModule.resolveSpeculation(
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
        XCTAssertEqual(served, Set(QWModelDescriptor.Family.allCases))
        XCTAssertEqual(unserved, [])
        for family in unserved {
            XCTAssertNil(QWModuleRegistry.moduleType(for: family), family.rawValue)
        }
    }

    func testTheQwen3FamiliesExposeUnifiedDenseCapabilities() throws {
        XCTAssertEqual(QWModuleRegistry.capabilities(for: .qwen3Dense), DenseModule.capabilities)
        XCTAssertEqual(QWModuleRegistry.capabilities(for: .qwen3MoE), DenseModule.capabilities)
        XCTAssertEqual(ObjectIdentifier(try XCTUnwrap(QWModuleRegistry.moduleType(for: .qwen3Dense))),
                       ObjectIdentifier(DenseModule.self))
        XCTAssertEqual(ObjectIdentifier(try XCTUnwrap(QWModuleRegistry.moduleType(for: .qwen3MoE))),
                       ObjectIdentifier(DenseModule.self))
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
        let modes = QWModuleRegistry.supportedKVStorageModes(for: flashNextDescriptor(advertising: [.mtp]))
        XCTAssertTrue(modes.contains("f32"))
        XCTAssertTrue(modes.contains("q4"))
        XCTAssertEqual(modes, DenseModule.supportedKVStorageModes(for: flashNextDescriptor(advertising: [.mtp])))
        XCTAssertFalse(QWModuleRegistry.supportedKVStorageModes(for: .qwen38_27b).contains("f32"))
    }

    func testCapabilitiesSayWhatEachEngineCanDo() {
        let dense = DenseModule.capabilities
        let flash = QWModelCapabilities.resolve(descriptor: flashNextDescriptor(advertising: [.mtp, .vision]),
                                                hasVisionWeights: true)
        XCTAssertTrue(dense.contains(.dflash), "Dense supports DFlash drafters")
        XCTAssertTrue(flash.contains(.dflash), "HyperStream supports DFlash 2 drafters")
        XCTAssertTrue(dense.contains(.ropeYaRN))
        XCTAssertFalse(flash.contains(.ropeYaRN), "HyperStream has no rope-YaRN path")
        XCTAssertFalse(flash.contains(.gdnPrefill),
                       "HyperStream echoes gdnPrefill, it does not choose it")
        XCTAssertTrue(dense.contains(.gdnPrefill))
    }

    func testFlashNextAndDenseSupportVision() {
        let flash = QWModelCapabilities.resolve(descriptor: flashNextDescriptor(advertising: [.vision]),
                                                hasVisionWeights: true)
        XCTAssertTrue(flash.contains(.vision),
                      "HyperStream (qwen4_exp) supports the 27-block ViT vision tower")
        XCTAssertTrue(DenseModule.capabilities.contains(.vision),
                      "the dense pack advertises a vision tower")
    }

    func testAdaptiveMTPIsDenseOnly() {
        let flash = QWModelCapabilities.resolve(descriptor: flashNextDescriptor(advertising: [.mtp]))
        XCTAssertTrue(DenseModule.capabilities.contains(.mtpAdaptive))
        XCTAssertFalse(flash.contains(.mtpAdaptive))
        XCTAssertTrue(flash.contains(.speculative))
    }

    func testEveryFamilyDeclaresACapabilitySetAndADoorSet() throws {
        for family in QWModuleRegistry.supportedFamilies {
            let type = try XCTUnwrap(QWModuleRegistry.moduleType(for: family))
            XCTAssertFalse(type.capabilities.isEmpty, family.rawValue)
            XCTAssertFalse(type.requiredDoors.isEmpty, family.rawValue)
            XCTAssertFalse(type.supportedKVStorageModes.isEmpty, family.rawValue)
        }
    }

    func testDoorsAreDistinctWithinAModule() {
        XCTAssertEqual(Set(DenseModule.requiredDoors).count, DenseModule.requiredDoors.count,
                       "a repeated door would prewarm twice")
        let hyperDoors = DenseModule.requiredDoors(for: flashNextDescriptor(advertising: [.mtp]))
        XCTAssertEqual(Set(hyperDoors).count, hyperDoors.count)
        let stdDoors = DenseModule.requiredDoors(for: .qwen38_27b)
        XCTAssertEqual(Set(stdDoors).count, stdDoors.count)
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
        XCTAssertTrue(module is DenseModule)
    }

    func testGeometryDerivedCapabilitiesPreserveUniversalAndTraitKnobs() throws {
        let fnVisionPlan = try flashNextPlan(advertising: [.vision, .mtp])
        XCTAssertTrue(try QWModuleRegistry.makeModule(plan: fnVisionPlan, engine: nil) is DenseModule)

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
        XCTAssertTrue(try QWModuleRegistry.makeModule(plan: stdVisionPlan, engine: nil) is DenseModule)

        let noVision = QWModelCapabilities.resolve(descriptor: stdDesc, hasVisionWeights: false)
        XCTAssertTrue(noVision.isSuperset(of: [.prefixCache, .kvQuantQ8, .kvQuantQ4, .lsa, .speculative, .dflash, .ropeYaRN, .mtpAdaptive]))
        XCTAssertFalse(noVision.contains(.vision))
        XCTAssertFalse(noVision.contains(.gdnPrefill))
        XCTAssertFalse(noVision.contains(.expertStreaming))

        let withVision = QWModelCapabilities.resolve(descriptor: stdDesc, hasVisionWeights: true)
        XCTAssertTrue(withVision.contains(.vision))
        XCTAssertEqual(QWModuleRegistry.capabilities(for: stdDesc, hasVisionWeights: true), withVision)

        let gdnHybrid = QWModelCapabilities.resolve(descriptor: .qwen38_27b, hasVisionWeights: false)
        XCTAssertTrue(gdnHybrid.contains(.gdnPrefill))

        let fnCaps = QWModelCapabilities.resolve(
            descriptor: flashNextDescriptor(advertising: [.vision, .mtp, .moe, .hyperConnections]),
            hasVisionWeights: true
        )
        XCTAssertEqual(fnCaps, [.prefixCache, .kvQuantQ8, .kvQuantQ4, .lsa, .speculative, .dflash, .vision])

        let hugeVocabDesc = QWModelDescriptor(
            family: .qwen3Dense, modelType: "qwen3", capabilities: [],
            hidden: 2560, vocab: 400_000,
            denseIntermediate: 9728, numLayers: 2,
            layerTypes: ["full_attention", "full_attention"],
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 32768,
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
        XCTAssertFalse(QWModelCapabilities.resolve(descriptor: hugeVocabDesc).contains(.dflash))

        let drafterDesc = QWModelDescriptor(
            family: .qwen3Dense, modelType: "dflash_drafter", capabilities: [],
            hidden: 2560, vocab: 151936,
            denseIntermediate: 9728, numLayers: 2,
            layerTypes: ["full_attention", "full_attention"],
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 32768,
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
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 151645,
            role: .speculativeDrafter
        )
        XCTAssertEqual(QWModelCapabilities.resolve(descriptor: drafterDesc, hasVisionWeights: true), [])
    }

    func testAdvertisedCapabilitiesMapToTheKnobCapabilitiesTheyNeed() {
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.vision]), [.vision])
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.mtp]), [.speculative])
        XCTAssertEqual(QWModuleRegistry.requiredCapabilities(advertised: [.moe, .ple]), [])
    }

    func testModuleStubsFailLoudRatherThanPretendingToWork() throws {
        let plan = try densePlan()
        let module = DenseModule(plan: plan)
        XCTAssertThrowsError(
            try module.generate(
                request: QWGenerationRequest(prompt: [1, 2], maxNew: 1, sampling: .greedy),
                onToken: nil
            )
        ) { error in
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
        let flashModule = DenseModule(plan: plan)
        let unified = try QWEngine(engine: flashModule, descriptor: desc, plan: plan)
        XCTAssertEqual(unified.plan.modelId, "qwen3.8-flash-next")
        XCTAssertEqual(unified.plan.layers.count, 4)
        XCTAssertTrue(unified.plan.layers.allSatisfy { $0.mlp == .moe })
        XCTAssertEqual(unified.plan.layers[0].attention, .gdn)
        XCTAssertEqual(unified.plan.layers[3].attention, .qsaSparse)
        XCTAssertEqual(unified.batchScheduler.inFlightRequests, 0)
        XCTAssertTrue(unified.supportsTokenConstraints)
    }

    func testMaxwellWeightAllocationPlanMTPAndQuantFormats() throws {
        let cfg = """
        {
          "model_type": "maxwell_moe",
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "vocab_size": 200704,
          "num_hidden_layers": 6,
          "mlp_only_layers": [0, 1, 2],
          "num_experts": 16,
          "num_experts_per_tok": 4,
          "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 1024,
          "use_shared_expert_gate": false,
          "moe_router_enable_expert_bias": true,
          "router_score_function": "sigmoid",
          "num_attention_heads": 32,
          "num_key_value_heads": 4,
          "attention_head_size": 128,
          "qk_layernorm": true,
          "attention_variant": "block_sparse_gqa",
          "block_sparse_gqa_indexer_n_heads": 4,
          "block_sparse_gqa_indexer_head_dim": 128,
          "block_sparse_gqa_block_size": 128,
          "block_sparse_gqa_prefix_window_size": 1024,
          "block_sparse_gqa_local_window_size": 1024,
          "block_sparse_gqa_num_remote_blocks": 16,
          "enable_hyper_connections": true,
          "num_residual_streams": 4,
          "mhc_sinkhorn_iterations": 20,
          "mtp_num_layers": 1,
          "rms_norm_eps": 1e-6,
          "rope_theta": 20000000,
          "max_position_embeddings": 65536,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfg)
        let geo = QWModelGeometry(descriptor: desc)
        let plan0 = try QWWeightAllocationPlan(geometry: geo, ropeRows: 8192, mtpDrafts: 0, format: .hq4G32)
        let plan1 = try QWWeightAllocationPlan(geometry: geo, ropeRows: 8192, mtpDrafts: 1, format: .hq4G32)
        XCTAssertGreaterThan(plan1.bytes, plan0.bytes)
        let shvqFormat = QWWeightFormat.hqBlock(
            bits: 8, groupSize: 32, expertBits: 4, expertGroupSize: 64,
            denseIncumbent: false, expertIncumbent: false,
            expertDownBits: 4, expertDownGroupSize: 64, expertDownIncumbent: false,
            denseSHVQ: false, expertSHVQ: true, expertDownSHVQ: true)
        let planShvq = try QWWeightAllocationPlan(geometry: geo, ropeRows: 8192, mtpDrafts: 1, format: shvqFormat)
        XCTAssertGreaterThan(planShvq.bytes, 0)
    }
}

private extension QWModuleRegistry {
    static func requiredCapabilities(
        advertised: Set<QWModelDescriptor.Capability>) -> Set<QWKnobCapability> {
        var required: Set<QWKnobCapability> = []
        if advertised.contains(.vision) { required.insert(.vision) }
        if advertised.contains(.mtp) { required.insert(.speculative) }
        return required
    }

    static func makeModule(
        plan: QWEnginePlan, engine: (any QWLoadedEngine)?
    ) throws -> any QWInferenceModule {
        guard let type = moduleType(for: plan.family) else {
            throw QWModuleError.unsupportedFamily(plan.family)
        }
        for capability in requiredCapabilities(advertised: plan.modelCapabilities) {
            guard type.capabilities.contains(capability) else {
                throw QWModuleError.unsupportedCapability(model: plan.modelId, capability: capability)
            }
        }
        return try type.make(plan: plan, engine: engine)
    }
}
