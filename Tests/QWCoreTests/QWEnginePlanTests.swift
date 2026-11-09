import XCTest
@testable import QWCore

/// QWEnginePlan freezes a model's numeric shape so both engines agree on it.
/// Every case runs on stub configs — the unit suite must not need an
/// installed model pack.
final class QWEnginePlanTests: XCTestCase {
    /// A dense 27B-shaped descriptor, so digest sensitivity can be tested
    /// without touching a 20 GB pack.
    private func denseDescriptor(hidden: Int = 5120, ropeTheta: Double = 1e7) -> QWModelDescriptor {
        let numLayers = 64
        let interval = 4
        let types = (0..<numLayers).map { (l: Int) -> String in
            (l + 1) % interval == 0 ? "full_attention" : "linear_attention"
        }
        return QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5",
            capabilities: [.vision, .mtp],
            hidden: hidden, vocab: 248320,
            denseIntermediate: 17408,
            numLayers: numLayers, layerTypes: types,
            fullAttentionInterval: interval,
            rmsEps: 1e-6,
            maxPositionEmbeddings: 262144,
            qHeads: 24, kvHeads: 4, headDim: 256,
            attnOutputGate: true,
            partialRotaryFactor: 0.25,
            ropeTheta: ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 16, gdnVHeads: 48,
            gdnHeadDim: 128, gdnQKVChannels: 10240,
            gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1,
            mtpUseDedicatedEmbeddings: false,
            eosTokenId: 248044)
    }

    /// A Flash-Next-shaped descriptor whose expert geometry is missing.
    private func moeDescriptorWithoutExpertWidths() -> QWModelDescriptor {
        QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.moe],
            hidden: 2560, vocab: 248320,
            denseIntermediate: nil,
            numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4,
            rmsEps: 1e-6,
            maxPositionEmbeddings: 262144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: true,
            partialRotaryFactor: 0.25,
            ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 512, expertsPerTok: 10, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 16, gdnVHeads: 48,
            gdnHeadDim: 128, gdnQKVChannels: 10240,
            gdnZChannels: 6144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0,
            mtpUseDedicatedEmbeddings: false,
            eosTokenId: 248044)
    }

    func testDensePlanDerivesWidthsFromGeometry() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "dense-27b",
                                    geometryDigest: "test", prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(plan.layers.count, geometry.numLayers)
        XCTAssertEqual(plan.qProjOut, geometry.qProjOut)
        XCTAssertEqual(plan.kvWidth, geometry.kvChannels)
        XCTAssertEqual(plan.attnOutWidth, geometry.qChannels)
        XCTAssertEqual(plan.prefillChunkRows, 512)
        XCTAssertEqual(plan.kvStepTokens, QWCapacityPlan.kvStepTokens)
        XCTAssertEqual(plan.nativeWindow, geometry.maxPositionEmbeddings)
        // 27B: 64 layers, every 4th is full attention, the rest are linear.
        XCTAssertEqual(plan.layers.filter { $0.attention == .fullAttention }.count, 16)
        XCTAssertEqual(plan.layers.filter { $0.attention == .gdn }.count, 48)
        XCTAssertTrue(plan.layers.allSatisfy { $0.mlp == .dense })
        XCTAssertFalse(plan.layers.contains { $0.hasPLE })
    }

    func testOutputGateWidensQProjOutButNotAttentionOutput() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        // q_proj packs Q‖gate; the attention block still emits qHeads * headDim.
        XCTAssertTrue(geometry.attnOutputGate)
        XCTAssertEqual(plan.qProjOut, 2 * geometry.qHeads * geometry.headDim)
        XCTAssertEqual(plan.attnOutWidth, geometry.qHeads * geometry.headDim)
    }

    func testFlashNextPlanHasQSASparseAndMoE() throws {
        let geometry = try QWStubModelConfig.flashNextGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "flash-next",
                                    geometryDigest: "test", prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(plan.layers.count, 48)
        XCTAssertEqual(plan.layers.filter { $0.attention == .qsaSparse }.count, 12)
        XCTAssertEqual(plan.layers.filter { $0.attention == .gdn }.count, 36)
        XCTAssertTrue(plan.layers.allSatisfy { $0.mlp == .moe })
        XCTAssertEqual(plan.kvWidth, 512)   // 2 KV heads x 256 — NOT the dense model's 1024
        XCTAssertEqual(plan.expertsPerToken, 10)
        XCTAssertEqual(plan.numExperts, 512)
        XCTAssertEqual(plan.moeIntermediate, 640)
        XCTAssertEqual(plan.sharedExpertIntermediate, 640)
    }

    func testPLEIsCarriedPerLayerNotPerModel() throws {
        let geometry = try QWStubModelConfig.flashNextGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        let expected = Set(geometry.descriptor.pleModuleIndices)
        XCTAssertFalse(expected.isEmpty, "the Flash-Next shape carries a PLE injection")
        for layer in plan.layers {
            XCTAssertEqual(layer.hasPLE, expected.contains(layer.index),
                           "layer \(layer.index) PLE flag")
        }
        XCTAssertLessThan(plan.layers.filter(\.hasPLE).count, plan.layers.count,
                          "PLE is not on every layer")
    }

    func testLayerIndicesAreContiguousAndOrdered() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(plan.layers.map(\.index), Array(0..<geometry.numLayers))
        for layer in plan.layers {
            XCTAssertEqual(layer.attention == .fullAttention, geometry.isFullAttention(layer.index))
        }
    }

    func testDigestIsStableAcrossConstruction() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let a = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d", prefillChunkRows: 512, kvStorageMode: "f16")
        let b = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d", prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(a.digest, b.digest)
        let c = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d", prefillChunkRows: 1024, kvStorageMode: "f16")
        XCTAssertNotEqual(a.digest, c.digest)
        let d = try QWEnginePlan(geometry: geometry, modelId: "other", geometryDigest: "d", prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertNotEqual(a.digest, d.digest)
    }

    func testDigestSeparatesModelIdFromGeometryDigest() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let a = try QWEnginePlan(geometry: geometry, modelId: "x|y", geometryDigest: "z",
                                  prefillChunkRows: 512, kvStorageMode: "f16")
        let b = try QWEnginePlan(geometry: geometry, modelId: "x", geometryDigest: "y|z",
                                  prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertNotEqual(a.digest, b.digest,
                          "a delimiter in a free-form field must not alias another pair")
    }

    func testDigestChangesWhenAnyGeometryFieldChanges() throws {
        let base = QWModelGeometry(descriptor: denseDescriptor())
        let plan = try QWEnginePlan(geometry: base, modelId: "m", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        let otherHidden = QWModelGeometry(descriptor: denseDescriptor(hidden: 4096))
        let otherRope = QWModelGeometry(descriptor: denseDescriptor(ropeTheta: 5e6))
        for variant in [otherHidden, otherRope] {
            let changed = try QWEnginePlan(geometry: variant, modelId: "m", geometryDigest: "d",
                                           prefillChunkRows: 512, kvStorageMode: "f16")
            XCTAssertNotEqual(plan.digest, changed.digest,
                              "the digest must bind the geometry, not just the caller's string")
        }
    }

    func testDigestBindsTheKVStorageMode() throws {
        // The KV mode selects the attention kernels, so two plans that differ
        // only in it must not share a digest.
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let f16 = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                   prefillChunkRows: 512, kvStorageMode: "f16")
        let q4 = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                  prefillChunkRows: 512, kvStorageMode: "q4")
        XCTAssertNotEqual(f16.digest, q4.digest)
        XCTAssertEqual(f16.kvStorageMode, "f16")
        XCTAssertEqual(q4.kvStorageMode, "q4")
    }

    func testEmptyKVStorageModeIsRejected() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        XCTAssertThrowsError(try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                               prefillChunkRows: 512, kvStorageMode: "")) { error in
            XCTAssertEqual(error as? QWEnginePlanError, .invalidKVStorageMode(""))
        }
    }

    func testPlanCarriesTheAdvertisedCapabilitiesAndTheFamily() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        let plan = try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                    prefillChunkRows: 512, kvStorageMode: "f16")
        XCTAssertEqual(plan.family, .qwen38Dense)
        XCTAssertEqual(plan.modelCapabilities, geometry.descriptor.capabilities)
        XCTAssertTrue(plan.modelCapabilities.contains(.mtp))
    }

    func testNonPositivePrefillChunkRowsIsRejected() throws {
        let geometry = try QWStubModelConfig.dense27BGeometry()
        XCTAssertThrowsError(try QWEnginePlan(geometry: geometry, modelId: "m", geometryDigest: "d",
                                               prefillChunkRows: 0, kvStorageMode: "f16")) { error in
            XCTAssertEqual(error as? QWEnginePlanError, .invalidPrefillChunkRows(0))
        }
    }

    func testMoEModelWithoutExpertGeometryFailsLoud() throws {
        let descriptor = moeDescriptorWithoutExpertWidths()
        XCTAssertThrowsError(try QWEnginePlan(geometry: QWModelGeometry(descriptor: descriptor),
                                               modelId: "m", geometryDigest: "d",
                                               prefillChunkRows: 512, kvStorageMode: "f16")) { error in
            XCTAssertEqual(error as? QWEnginePlanError, .incompleteMoEGeometry)
        }
    }
}
