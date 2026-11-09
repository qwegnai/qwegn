// ToyModule.swift — A third module, added without touching a production file.
//
// The proof: a model shape the engine has never served — four full-attention
// layers, no GDN at all, an f32 KV cache the dense model does not have —
// plugs into the same seam, resolves its layer kernels through the same
// registry, and serves tokens.

import Foundation
import QWCore
import QWHead
@testable import QWEngine

/// A module for a model shape the engine has never served.
/// It runs no kernels: it proves that the seam, the plan and the layer
/// registry are everything a new model needs. Its family is the dense one.
final class ToyModule: QWInferenceModule, @unchecked Sendable {
    static let family: QWModelDescriptor.Family = .qwen38Dense
    static let capabilities: Set<QWKnobCapability> = []

    /// The toy runs an f32 KV cache — a mode the dense model does not have.
    static let supportedKVStorageModes: Set<String> = ["f32"]
    static let requiredDoors: [String] = []

    /// The module's own session state.
    final class State: QWModuleSessionState, @unchecked Sendable {
        var activeTokens: Int = 0
        /// The tokens the module has been fed, in order.
        var history: [Int] = []

        func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws {
            activeTokens = 0
            history = []
        }
    }

    let plan: QWEnginePlan

    /// The kernel each layer of the plan runs, resolved once at init.
    let layerKernels: [any QWLayerKernel]

    /// The toy reports no cache events, but it accepts the sink.
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?

    init(plan: QWEnginePlan) throws {
        self.plan = plan
        self.layerKernels = try QWLayerKernelRegistry.resolve(plan: plan)
    }

    /// Builds the toy module and the plan it serves.
    static func make(hidden: Int = 64, layers: Int = 4,
                     kvStorageMode: String = "f32") throws -> ToyModule {
        try ToyModule(plan: try makePlan(hidden: hidden, layers: layers,
                                         kvStorageMode: kvStorageMode))
    }

    /// The plan of a toy model: `layers` full-attention layers and no GDN.
    static func makePlan(hidden: Int, layers: Int, kvStorageMode: String,
                          capabilities: Set<QWModelDescriptor.Capability> = []) throws -> QWEnginePlan {
        let descriptor = QWModelDescriptor(
            family: .qwen38Dense, modelType: "toy", capabilities: capabilities,
            hidden: hidden, vocab: 32, denseIntermediate: hidden * 2, numLayers: layers,
            layerTypes: Array(repeating: "full_attention", count: layers),
            fullAttentionInterval: layers, rmsEps: 1e-6, maxPositionEmbeddings: 2048,
            qHeads: 4, kvHeads: 2, headDim: 16, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 10000,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0,
            gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 0)
        let geometry = QWModelGeometry(descriptor: descriptor)
        return try QWEnginePlan(geometry: geometry, modelId: "toy-third-module",
                                geometryDigest: "toy", prefillChunkRows: 8,
                                kvStorageMode: kvStorageMode)
    }

    static func make(plan: QWEnginePlan, engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        try ToyModule(plan: plan)
    }

    /// The toy module builds no engine: it proves the seam, not a model.
    static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        throw QWModuleError.notImplemented("ToyModule.makeEngine")
    }

    static func resolveSpeculation(
        _ parameters: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        QWResolvedSpeculation(parameters: parameters, engineDrafts: 2, engineBackend: nil,
                              reportedDrafts: 2, reportedBackend: "toy")
    }

    var kvStorageMode: String { plan.kvStorageMode }

    var hasPrefixCache: Bool { false }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        throw QWModuleError.notImplemented("ToyModule.configurePrefixCache")
    }

    func makeSessionState() throws -> any QWModuleSessionState {
        State()
    }

    func prepare(_ state: any QWModuleSessionState, prompt: [Int], maxNew: Int) throws {
        // The toy has no stack to carve: the plan already says how much a
        // request may generate.
    }

    func finish(_ state: any QWModuleSessionState) {
        // The toy holds nothing across a request.
    }

    func encodePrefill(_ state: any QWModuleSessionState, tokens: [Int],
                       chunk: Range<Int>, runsHead: Bool) throws -> Int {
        guard let toy = state as? State else {
            throw QWModuleError.invalidState("ToyModule got a foreign session state")
        }
        for token in tokens[chunk] {
            toy.history.append(token)
            toy.activeTokens += 1
        }
        guard runsHead, let last = chunk.last else { return 0 }
        return step(toy, token: last)
    }

    func encodeDecodeStep(_ state: any QWModuleSessionState, token: Int,
                          position: Int) throws -> Int {
        guard let toy = state as? State else {
            throw QWModuleError.invalidState("ToyModule got a foreign session state")
        }
        toy.history.append(token)
        toy.activeTokens += 1
        return step(toy, token: token)
    }

    /// The module's own numerics: a toy recurrence over the layers the plan
    /// names. Each layer scales the accumulator by its own kind's weight, so
    /// the plan's layers — and their number — decide the output.
    func step(_ state: State, token: Int) -> Int {
        var accumulator = Double(token) + Double(state.history.count)
        for (index, kernel) in layerKernels.enumerated() {
            let weight = type(of: kernel).key.attention == .gdn ? 0.5 : 1.0
            accumulator = accumulator * (1 + weight) + Double(index + 1)
        }
        return Int(abs(accumulator).rounded(.down)) % max(1, plan.vocab)
    }

    func lastStepMilliseconds(_ state: any QWModuleSessionState) -> Double { 0 }

    func prefillGPUSeconds(_ state: any QWModuleSessionState) -> Double { 0 }

    func sampleLogits(_ state: any QWModuleSessionState) throws -> [Float] {
        // The toy has no logits. It refuses rather than answering with a row
        // of zeros — a numeric zero is not "unavailable".
        throw QWModuleError.notImplemented("ToyModule.sampleLogits")
    }
}
