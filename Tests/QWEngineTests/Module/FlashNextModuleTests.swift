import XCTest
import QWCore
import QWHead
@testable import QWEngine

/// The unified sampler: one implementation, both families. These tests use
/// no model pack — a pack is a runtime artefact, never a test fixture.
final class FlashNextModuleTests: XCTestCase {
    func testDraftStrategyResolutionAndReporting() throws {
        let requested = QWSpeculativeParameters(mtpDrafts: 5, mtpBackend: "variational",
                                                mtpDraftStrategy: "hybrid")
        let resolved = try FlashNextModule.resolveSpeculation(
            requested, descriptor: .qwen38_27b,
            environment: ["FN_DRAFT": "NeUrAl"], modelSwitch: false)
        XCTAssertEqual(resolved.reportedBackend, "mtp")
        XCTAssertEqual(resolved.reportedDrafts, 3)
        XCTAssertEqual(resolved.parameters.mtpDraftStrategy, "mtp")
        XCTAssertEqual(try FNGPUSessionState.draftMode(
            for: resolved.parameters, environment: ["FN_DRAFT": "ngram"]), .mtp)
        XCTAssertEqual(try FNGPUSessionState.draftMode(for: requested, environment: ["FN_DRAFT": "ngram"]), .ngram)
        XCTAssertThrowsError(try FlashNextModule.resolveSpeculation(
            requested, descriptor: .qwen38_27b, environment: ["FN_DRAFT": "typo"], modelSwitch: false))
        var invalid = requested
        invalid.mtpDraftStrategy = "typo"
        XCTAssertThrowsError(try FlashNextModule.resolveSpeculation(
            invalid, descriptor: .qwen38_27b, environment: [:], modelSwitch: false))
        invalid = requested
        invalid.mtpBackend = "typo"
        XCTAssertThrowsError(try FlashNextModule.resolveSpeculation(
            invalid, descriptor: .qwen38_27b, environment: [:], modelSwitch: false))
    }

    /// The repository root, from this file's own path.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // QWEngineTests/Module
            .deletingLastPathComponent()   // QWEngineTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
    }

    func testBothFamiliesSampleTheSameTokenFromTheSameLogits() throws {
        // The same seed and the same logits must pick the same token, so a
        // request means the same thing on either engine.
        let logits: [Float] = [0.1, 5.0, 0.2, 0.3]
        let config = QWSamplerConfig(temperature: 1.0, topK: 2, topP: 1.0, seed: 42)
        let dense = QWSampler(config: config)
        let flash = QWSampler(config: config)
        XCTAssertEqual(try dense.sample(logits: logits).token,
                       try flash.sample(logits: logits).token)
    }

    func testTheSharedSamplerAdvancesItsDrawCounter() throws {
        // A static entry point would draw the same uniform every call and
        // stop being random; the counter is what makes it a sequence.
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1.0, topK: 4,
                                                       topP: 1.0, seed: 7))
        XCTAssertEqual(sampler.drawsConsumed, 0)
        _ = try sampler.sample(logits: [0.1, 5.0, 0.2, 0.3])
        XCTAssertEqual(sampler.drawsConsumed, 1)
        _ = try sampler.sample(logits: [0.1, 5.0, 0.2, 0.3])
        XCTAssertEqual(sampler.drawsConsumed, 2)
    }

    func testTheSamplerIsReproducibleForAGivenSeed() throws {
        let logits: [Float] = [0.1, 5.0, 0.2, 0.3, -1.0, 0.9]
        let config = QWSamplerConfig(temperature: 0.8, topK: 3, topP: 0.9, seed: 99)
        let first = try QWSampler(config: config).sample(logits: logits).token
        let second = try QWSampler(config: config).sample(logits: logits).token
        XCTAssertEqual(first, second)
    }

    func testTheGreedyTokenIsTheArgmax() throws {
        let logits: [Float] = [0.1, 5.0, 0.2, 0.3]
        XCTAssertEqual(try QWSampler.greedyToken(logits: logits), 1)
    }

    func testTheRepetitionPenaltyShiftsTheGreedyToken() throws {
        // The penalty pushes a repeated token's logit down, so the greedy
        // pick moves to the next best one: 5.0 / 4.0 = 1.25 loses to 4.8.
        let logits: [Float] = [0.1, 5.0, 0.2, 4.8]
        XCTAssertEqual(try QWSampler.greedyToken(logits: logits), 1)
        XCTAssertEqual(try QWSampler.greedyToken(logits: logits,
                                                 repetitionHistory: [1],
                                                 repetitionPenalty: 4.0), 3)
    }

    func testFlashNextHasNoTokenLoopOfItsOwn() throws {
        // C2: one token loop. FN's private loop is gone; the core drives it.
        let source = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Subsystems/QWFlashNext/GPU/FNGPUEngine.swift"),
            encoding: .utf8)
        XCTAssertFalse(source.contains("func generate"),
                       "FN must call the core's loop, not keep one of its own")
        XCTAssertFalse(source.contains("while generated.count < maxNew"),
                       "FN must call the core's loop, not keep one of its own")
    }

    func testFlashNextHasNoGlobalStateResetLeftInTheEngine() throws {
        // C5: one session model. The recurrent state belongs to the session,
        // not to a global operation the engine runs on every request.
        let source = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Subsystems/QWFlashNext/GPU/FNGPUEngine.swift"),
            encoding: .utf8)
        XCTAssertFalse(source.contains("resetState"),
                       "the global reset must not stay in the engine")
    }

    func testTheFacadeOverTheCoreLivesInTheRuntimeLayer() throws {
        // The engine owns the numerics; the loop belongs to the core, which
        // sits above the subsystem. So the façade is an extension there.
        let source = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Module/FlashNextGeneration.swift"),
            encoding: .utf8)
        XCTAssertTrue(source.contains("extension FNGPUEngine"),
                      "the façade is the runtime's extension on the engine")
        XCTAssertTrue(source.contains("QWEngineCore"),
                      "the façade must drive the core's loop")
    }

    func testFlashNextHasNoSamplerOfItsOwn() throws {
        // FN must go through the shared sampler, not a private top-k/top-p.
        let engine = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Subsystems/QWFlashNext/GPU/FNGPUEngine.swift"),
            encoding: .utf8)
        XCTAssertFalse(engine.contains("func sampleToken("),
                       "FN must use the shared sampler, not its own")
        XCTAssertFalse(engine.contains("func sampleTopKTopP("),
                       "FN must use the shared sampler, not its own")
        // The shared sampler is built where the session is handed out — the
        // module, which sits above the subsystem — and the engine only
        // draws from it.
        let module = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Module/FlashNextModule.swift"),
            encoding: .utf8)
        XCTAssertTrue(module.contains("QWSampler("),
                      "the session must be handed the shared sampler")
        XCTAssertTrue(engine.contains("sampler: QWSampler"),
                      "the engine's sampler must be the shared one")
    }

    func testAModuleThatCannotSampleRefusesRatherThanAnsweringWithZeros() throws {
        // The seam hands the shared sampler a row; a module that cannot
        // sample has no row to give. A row of zeros is not "unavailable".
        let plan = try QWEnginePlan(geometry: QWModelGeometry.qwen38_27b, modelId: "toy",
                                    geometryDigest: "d", prefillChunkRows: 512,
                                    kvStorageMode: "f16")
        let module = LoopToyModule(plan: plan)
        let state = try module.makeSessionState()
        XCTAssertThrowsError(try module.sampleLogits(state)) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
    }

    func testTheSharedEntryPointAdvancesTheDrawCounter() throws {
        // `sampleToken` is the entry point FN uses. If it were static it
        // would draw the same uniform on every call — and this test is the
        // one that would catch it.
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1.0, topK: 4,
                                                        topP: 1.0, seed: 7))
        XCTAssertEqual(sampler.drawsConsumed, 0)
        let logits: [Float] = [0.1, 5.0, 0.2, 0.3]
        _ = try sampler.sampleToken(logits: logits)
        XCTAssertEqual(sampler.drawsConsumed, 1)
        _ = try sampler.sampleToken(logits: logits)
        XCTAssertEqual(sampler.drawsConsumed, 2)
    }

    func testBothFamiliesBuildTheSameContractFromTheSameRequest() throws {
        // The claim is that one request means one thing on either family.
        // FN builds its contract through the bridge; the dense path builds
        // the same contract by hand. They must agree, draw for draw.
        let request = QWSamplingParameters(temperature: 0.8, topP: 0.9, topK: 40,
                                           maxTokens: 32, repetitionPenalty: 1.05, seed: 5)
        let vocabulary = 1000
        let flashNext = QWSamplerConfig(request, vocabulary: vocabulary)
        let dense = QWSamplerConfig(temperature: Float(request.temperature),
                                    topK: request.topK > 0 ? request.topK : vocabulary,
                                    topP: Float(request.topP),
                                    seed: request.resolvedSeed(),
                                    uniformVersion: .upper24,
                                    repetitionPenalty: Float(request.repetitionPenalty))
        XCTAssertEqual(flashNext, dense)
        let logits = (0..<vocabulary).map { Float($0 % 17) / 10 }
        XCTAssertEqual(try QWSampler(config: flashNext).sampleToken(logits: logits),
                       try QWSampler(config: dense).sampleToken(logits: logits))
    }

    func testTheGreedyPathMatchesTheAlgorithmFlashNextUsedBefore() throws {
        // The byte-parity law for the greedy path. The reference below is
        // the pre-unification Flash-Next algorithm, verbatim.
        var seed: UInt64 = 0x5EED_1234
        func next() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(seed >> 40) / Float(1 << 24)
        }
        for _ in 0..<200 {
            let count = 2 + Int(next() * 12)
            var logits: [Float] = (0..<count).map { _ in Float(next() * 4 - 2) }
            // Ties, signed zeros and repeated maxima are where an argmax
            // implementation drifts.
            if !logits.isEmpty { logits[0] = 0 }
            if count > 3 { logits[count - 1] = -0.0 }
            if count > 5 { logits[3] = logits[1] }
            let history: [Int64] = [0, 1, 1, 99, -3, Int64(count)]
            for penalty in [1.0 as Float, 1.05, 2.0, 4.0] {
                let legacy = legacyGreedyToken(logits: logits, history: history,
                                               repetitionPenalty: Double(penalty))
                let shared = try QWSampler.greedyToken(logits: logits,
                                                       repetitionHistory: history.compactMap { id in
                                                           (id >= 0 && id < count) ? Int(id) : nil
                                                       },
                                                       repetitionPenalty: penalty)
                XCTAssertEqual(legacy, shared, "penalty \(penalty) on \(logits)")
            }
        }
    }

    /// The greedy algorithm Flash-Next used before the sampler was shared.
    private func legacyGreedyToken(logits: [Float], history: [Int64],
                                    repetitionPenalty: Double) -> Int {
        let rep = Float(repetitionPenalty)
        var row = logits
        if rep != 1.0 && rep > 0 {
            var seen = Set<Int>()
            for token in history {
                let id = Int(token)
                if id >= 0 && id < row.count && seen.insert(id).inserted {
                    if row[id] > 0 { row[id] /= rep } else { row[id] *= rep }
                }
            }
        }
        var best = 0
        for index in 1..<row.count where row[index] > row[best] { best = index }
        return best
    }

    /// A Flash-Next-shaped descriptor, built in memory: a pack is a runtime
    /// artefact, never a test fixture.
    private func flashNextPlan() throws -> QWEnginePlan {
        let descriptor = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.mtp],
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
        return try QWEnginePlan(geometry: QWModelGeometry(descriptor: descriptor),
                                modelId: "qwen3.8-flash-next", geometryDigest: "d",
                                prefillChunkRows: 512, kvStorageMode: "f16")
    }

    func testTheFNModuleFailsLoudWithoutAnEngine() throws {
        let module = FlashNextModule(plan: try flashNextPlan())
        // A module with no engine must not pretend to serve.
        XCTAssertThrowsError(try module.makeSessionState()) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
        XCTAssertThrowsError(try module.sampleLogits(DummyState())) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
    }

    func testTheFNModuleRefusesAForeignEngine() throws {
        // A dense engine in Flash-Next's slot is an error, not a silent
        // mis-dispatch.
        let plan = try flashNextPlan()
        XCTAssertThrowsError(try FlashNextModule.make(plan: plan, engine: ForeignEngine(plan: plan))) { error in
            XCTAssertEqual(error as? QWModuleError, .unsupportedFamily(.qwen38FlashNext))
        }
    }

    func testTheFNModuleExistsWithoutAnEngine() throws {
        // No engine yet: the module exists and every call that needs the
        // engine fails loud.
        let plan = try flashNextPlan()
        let module = try FlashNextModule.make(plan: plan, engine: nil)
        XCTAssertTrue(module is FlashNextModule)
    }

    func testTheFNModuleCarriesItsPlanAndTypeFacts() throws {
        let plan = try flashNextPlan()
        let module = FlashNextModule(plan: plan)
        XCTAssertEqual(module.plan.digest, plan.digest)
        XCTAssertEqual(module.kvStorageMode, "f16")
        XCTAssertTrue(FlashNextModule.supportedKVStorageModes.contains("f32"))
        XCTAssertTrue(FlashNextModule.capabilities.contains(.vision))
        XCTAssertTrue(FlashNextModule.capabilities.contains(.speculative))
    }
}

/// A session state from another family — the module must not accept it.
private final class DummyState: QWModuleSessionState, @unchecked Sendable {
    var activeTokens: Int { 0 }
    func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws {}
}

/// A loaded engine from another family.
private final class ForeignEngine: QWLoadedEngine {
    let plan: QWEnginePlan
    init(plan: QWEnginePlan) { self.plan = plan }
}
