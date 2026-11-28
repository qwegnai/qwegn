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
        let resolved = try DenseModule.resolveSpeculation(
            requested, descriptor: flashNextDescriptor(),
            environment: ["FN_DRAFT": "NeUrAl"], modelSwitch: false)
        XCTAssertEqual(resolved.reportedBackend, "mtp")
        XCTAssertEqual(resolved.reportedDrafts, 3)
        XCTAssertEqual(resolved.parameters.mtpDraftStrategy, "mtp")
        XCTAssertEqual(try FNGPUSessionState.draftMode(
            for: resolved.parameters, environment: ["FN_DRAFT": "ngram"]), .mtp)
        XCTAssertEqual(try FNGPUSessionState.draftMode(for: requested, environment: ["FN_DRAFT": "ngram"]), .ngram)
        XCTAssertThrowsError(try DenseModule.resolveSpeculation(
            requested, descriptor: flashNextDescriptor(), environment: ["FN_DRAFT": "typo"], modelSwitch: false))
        var invalid = requested
        invalid.mtpDraftStrategy = "typo"
        XCTAssertThrowsError(try DenseModule.resolveSpeculation(
            invalid, descriptor: flashNextDescriptor(), environment: [:], modelSwitch: false))
        invalid = requested
        invalid.mtpBackend = "typo"
        XCTAssertThrowsError(try DenseModule.resolveSpeculation(
            invalid, descriptor: flashNextDescriptor(), environment: [:], modelSwitch: false))
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

    private static func readEngineSources() throws -> String {
        try [
            "Sources/Runtime/QWEngine/Driver/HyperStream/QWHyperStreamRuntime.swift",
            "Sources/Runtime/QWEngine/Driver/HyperStream/QWHyperStreamRuntime+Seam.swift",
        ].map {
            try String(contentsOf: Self.repoRoot.appendingPathComponent($0), encoding: .utf8)
        }.joined(separator: "\n")
    }

    func testFlashNextHasNoTokenLoopOfItsOwn() throws {
        // C2: one token loop. HyperStream's private loop is gone; the unified runtime drives it.
        let source = try Self.readEngineSources()
        XCTAssertFalse(source.contains("func generate"),
                       "QWHyperStreamRuntime must route through the core loop, not keep one of its own")
        XCTAssertFalse(source.contains("while generated.count < maxNew"),
                       "QWHyperStreamRuntime must route through the core loop, not keep one of its own")
    }

    func testFlashNextHasNoGlobalStateResetLeftInTheEngine() throws {
        // C5: one session model. The recurrent state belongs to the session,
        // not to a global operation the engine runs on every request.
        let source = try Self.readEngineSources()
        XCTAssertFalse(source.contains("resetState"),
                       "the global reset must not stay in the engine")
    }

    func testTheFacadeOverTheCoreLivesInTheRuntimeLayer() throws {
        for retired in [
            "Sources/Runtime/QWEngine/Module/FlashNextModule.swift",
            "Sources/Runtime/QWEngine/Module/FlashNextGeneration.swift",
            "Sources/Subsystems/QWFlashNext",
        ] {
            XCTAssertFalse(FileManager.default.fileExists(atPath: Self.repoRoot.appendingPathComponent(retired).path),
                           "\(retired) must be retired into unified QWEngineDriver/QWHyperStreamRuntime")
        }
        let conformance = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Generation/QWEngineDriverConformance.swift"),
            encoding: .utf8)
        XCTAssertTrue(conformance.contains("trunkExecutor.runGeneration("),
                      "QWEngineDriver conformance must route all trunks through QWTrunkExecutor")
        XCTAssertFalse(conformance.contains("hyperRuntime"),
                       "QWEngineDriver conformance must not name hyperRuntime")
        XCTAssertFalse(conformance.contains("inkling2Runtime"),
                       "QWEngineDriver conformance must not name inkling2Runtime")
        let generation = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Driver/HyperStream/QWHyperStreamRuntime+Generation.swift"),
            encoding: .utf8)
        XCTAssertTrue(generation.contains("extension QWHyperStreamRuntime"),
                      "HyperStream generation lives as an extension on QWHyperStreamRuntime")
        XCTAssertTrue(generation.contains("func runGeneration("),
                      "QWHyperStreamRuntime must expose the unified runGeneration path")
    }

    func testFlashNextHasNoSamplerOfItsOwn() throws {
        // HyperStream must go through the shared sampler, not a private top-k/top-p.
        let engine = try Self.readEngineSources()
        XCTAssertFalse(engine.contains("func sampleToken("),
                       "HyperStream must use the shared sampler, not its own")
        XCTAssertFalse(engine.contains("func sampleTopKTopP("),
                       "HyperStream must use the shared sampler, not its own")
        let generation = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Driver/HyperStream/QWHyperStreamRuntime+Generation.swift"),
            encoding: .utf8)
        let session = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Driver/HyperStream/FNGPUSession.swift"),
            encoding: .utf8)
        XCTAssertTrue(generation.contains("request.sampling"),
                      "generation must forward the shared sampler contract")
        XCTAssertTrue(session.contains("sampler: QWSampler"),
                      "HyperStream session's sampler must be the shared one")
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
    private func flashNextDescriptor() -> QWModelDescriptor {
        QWModelDescriptor(
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
    }

    private func flashNextPlan() throws -> QWEnginePlan {
        try QWEnginePlan(geometry: QWModelGeometry(descriptor: flashNextDescriptor()),
                         modelId: "qwen3.8-flash-next", geometryDigest: "d",
                         prefillChunkRows: 512, kvStorageMode: "f16")
    }

    func testTheFNModuleFailsLoudWithoutAnEngine() throws {
        let module = DenseModule(plan: try flashNextPlan())
        // A module with no engine must not pretend to serve.
        XCTAssertThrowsError(
            try module.generate(
                request: QWGenerationRequest(prompt: [1, 2], maxNew: 1, sampling: .greedy),
                onToken: nil
            )
        ) { error in
            XCTAssertEqual(error as? QWModuleError, .notImplemented("DenseModule has no engine yet"))
        }
        XCTAssertThrowsError(try module.configurePrefixCache(QWCacheConfig())) { error in
            guard case QWModuleError.notImplemented = error else {
                return XCTFail("expected notImplemented, got \(error)")
            }
        }
    }

    func testTheFNModuleRefusesAForeignEngine() throws {
        let plan = try flashNextPlan()
        XCTAssertThrowsError(try DenseModule.make(plan: plan, engine: ForeignEngine(plan: plan))) { error in
            XCTAssertEqual(error as? QWModuleError, .unsupportedFamily(.qwen38Dense))
        }
    }

    func testTheFNModuleExistsWithoutAnEngine() throws {
        let plan = try flashNextPlan()
        let module = try DenseModule.make(plan: plan, engine: nil)
        XCTAssertTrue(module is DenseModule)
    }

    func testTheFNModuleCarriesItsPlanAndTypeFacts() throws {
        let desc = flashNextDescriptor()
        let plan = try flashNextPlan()
        let module = DenseModule(plan: plan)
        XCTAssertEqual(module.plan.digest, plan.digest)
        XCTAssertEqual(module.kvStorageMode, "f16")
        XCTAssertTrue(DenseModule.supportedKVStorageModes(for: desc).contains("f32"))
        let caps = QWModelCapabilities.resolve(descriptor: desc, hasVisionWeights: true)
        XCTAssertTrue(caps.contains(.vision))
        XCTAssertTrue(caps.contains(.speculative))
        XCTAssertTrue(caps.contains(.dflash))
    }

    func testEstimateContextMemoryReflectsDFlashSavings() {
        let desc = QWModelDescriptor(
            family: .qwen38FlashNext, modelType: "qwen4_exp",
            capabilities: [.mtp, .vision],
            hidden: 2560, vocab: 248_320,
            denseIntermediate: nil, numLayers: 48,
            layerTypes: (0..<48).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" },
            fullAttentionInterval: 4, rmsEps: 1e-6, maxPositionEmbeddings: 262_144,
            qHeads: 24, kvHeads: 2, headDim: 256,
            attnOutputGate: false, partialRotaryFactor: 0.25, ropeTheta: 1e7,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: 256, expertsPerTok: 8, moeIntermediate: 640,
            sharedExpertIntermediate: 640, expertGateType: nil,
            hcCount: 4, hcLowrank: 320, gdnQKHeads: 16, gdnVHeads: 48, gdnHeadDim: 128,
            gdnQKVChannels: 10_240, gdnZChannels: 6_144, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 248_044
        )
        let missing = "/tmp/nonexistent-pack-\(UUID().uuidString)"
        let nativeMTP = DenseModule.estimateContextMemory(
            descriptor: desc, packPath: missing, contextTokens: 16_384, kvMode: .f16,
            mtpDrafts: 3, mtpBackend: "variational", prefillChunkSize: 4096,
            tuning: .default, physicalMemoryBytes: 64 << 30)
        let dflash = DenseModule.estimateContextMemory(
            descriptor: desc, packPath: missing, contextTokens: 16_384, kvMode: .f16,
            mtpDrafts: 7, mtpBackend: "dflash2", prefillChunkSize: 4096,
            tuning: .default, physicalMemoryBytes: 64 << 30)
        XCTAssertLessThan(dflash.kvBytes, nativeMTP.kvBytes)
        XCTAssertLessThan(dflash.weightsBytes, nativeMTP.weightsBytes)
        XCTAssertGreaterThan(dflash.maxFittingContext, nativeMTP.maxFittingContext)
    }

    func testResolveSpeculationDFlash2DSpineAndIncompatibleFallback() throws {
        let desc = flashNextDescriptor()
        func writeSyntheticDrafter(blockSize: Int, isDSpine: Bool) throws -> URL {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("fn_dflash_spec_\(UUID().uuidString)", isDirectory: true)
            let assets = dir.appendingPathComponent("assets")
            try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
            let dflashDict: [String: Any] = [
                "block_size": blockSize,
                "target_layer_ids": [0],
            ]
            var cfg: [String: Any] = [
                "model_type": "qwen3",
                "hidden_size": desc.hidden, "num_hidden_layers": 1, "vocab_size": desc.vocab,
                "num_attention_heads": 20, "num_key_value_heads": 8,
                "head_dim": 128, "dflash_config": dflashDict
            ]
            if isDSpine { cfg["selected_prefix_code_bridge"] = ["code_size": 64, "rank": 32] }
            try JSONSerialization.data(withJSONObject: cfg).write(to: assets.appendingPathComponent("config.json"))
            var tensors: [String: Any] = [
                "fc.weight": ["shape": [desc.hidden, desc.hidden], "hq4_off": 64],
                "hidden_norm.weight": ["shape": [desc.hidden], "raw_off": 64],
                "norm.weight": ["shape": [desc.hidden], "raw_off": 64],
                "layers.0.self_attn.q_proj.weight": ["shape": [desc.hidden, desc.hidden], "hq4_off": 64],
            ]
            if isDSpine {
                tensors["selected_prefix_bridge.codebook"] = ["shape": [64, 32], "raw_off": 64]
                tensors["selected_prefix_bridge.read_weight"] = ["shape": [32, desc.hidden], "raw_off": 64]
                tensors["selected_prefix_bridge.feature_write_weights"] = ["shape": [desc.hidden, 32], "raw_off": 64]
            }
            let hdrData = try JSONSerialization.data(withJSONObject: [
                "provenance": ["source": "unit-test"],
                "tensors": tensors,
            ])
            var qwp = Data("QWP1".utf8)
            var len = UInt64(hdrData.count)
            withUnsafeBytes(of: &len) { qwp.append(contentsOf: $0) }
            qwp.append(Data(count: 64))
            qwp.append(hdrData)
            try qwp.write(to: dir.appendingPathComponent("model.qwp"))
            return dir
        }

        let dflashDir = try writeSyntheticDrafter(blockSize: 8, isDSpine: false)
        defer { try? FileManager.default.removeItem(at: dflashDir) }
        var reqDFlash = QWSpeculativeParameters(mtpDrafts: 7, mtpBackend: "dflash2")
        reqDFlash.dflashDraftPackPath = dflashDir.path
        let resDF = try DenseModule.resolveSpeculation(reqDFlash, descriptor: desc, environment: [:], modelSwitch: true)
        XCTAssertEqual(resDF.engineBackend, QWMTPBackend.dflash2)
        XCTAssertEqual(resDF.engineDrafts, 7)
        XCTAssertEqual(resDF.reportedBackend, "dflash2")

        let dspineDir = try writeSyntheticDrafter(blockSize: 16, isDSpine: true)
        defer { try? FileManager.default.removeItem(at: dspineDir) }
        var reqDS = QWSpeculativeParameters(mtpDrafts: 12, mtpBackend: "dspine")
        reqDS.dflashDraftPackPath = dspineDir.path
        let resDS = try DenseModule.resolveSpeculation(reqDS, descriptor: desc, environment: [:], modelSwitch: true)
        XCTAssertEqual(resDS.engineBackend, QWMTPBackend.dflash2)
        XCTAssertEqual(resDS.engineDrafts, 7)
        XCTAssertEqual(resDS.reportedBackend, "dspine")

        // Model switch with incompatible/missing drafter falls back cleanly to variational (<= 3 drafts).
        var reqMissing = QWSpeculativeParameters(mtpDrafts: 7, mtpBackend: "dflash2")
        reqMissing.dflashDraftPackPath = "/tmp/nonexistent-dflash-\(UUID().uuidString)"
        let fallback = try DenseModule.resolveSpeculation(reqMissing, descriptor: desc, environment: [:], modelSwitch: true)
        XCTAssertEqual(fallback.parameters.mtpBackend, "variational")
        XCTAssertNil(fallback.engineBackend)
        XCTAssertLessThanOrEqual(fallback.reportedDrafts, 3)
    }

    func testConcurrentLastResultAndTelemetryUpdatesAreThreadSafe() throws {
        let genSource = try String(contentsOf: Self.repoRoot
            .appendingPathComponent("Sources/Runtime/QWEngine/Driver/HyperStream/QWHyperStreamRuntime+Generation.swift"),
            encoding: .utf8)
        XCTAssertTrue(genSource.contains("stateLock.withLock { _naxLogMessage = newValue }"))
        XCTAssertTrue(genSource.contains("stateLock.withLock { _lastResult = newValue }"))
        let module = DenseModule(plan: try flashNextPlan())
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "fn.module.race.test", attributes: .concurrent)
        for _ in 0..<8 {
            group.enter()
            queue.async {
                for _ in 0..<250 {
                    module.onCacheEvent = { _ in }
                    _ = module.lastTurn
                    _ = module.report
                    _ = module.capabilities
                    _ = module.prefixCache
                }
                group.leave()
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
        XCTAssertNil(module.lastTurn)
    }
}

/// A loaded engine from another family.
private final class ForeignEngine: QWLoadedEngine {
    let plan: QWEnginePlan
    init(plan: QWEnginePlan) { self.plan = plan }
}
