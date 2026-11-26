// QWDecisionPackEngineGeneralizationTests.swift — Tests for generalized 4-head, LayerNorm, GELU, and embedding execution (G1).

import Foundation
import Metal
import QWCore
import QWDecision
import QWEngine
import XCTest

final class QWDecisionPackEngineGeneralizationTests: XCTestCase {
    private func makeDescriptor(
        family: QWModelDescriptor.Family,
        modelType: String,
        head: QWDecisionModelConfig.HeadKind,
        topology: QWDecisionModelConfig.AttentionTopology,
        framing: QWDecisionModelConfig.SequenceFraming,
        normKind: NormKind?,
        activationKind: QWModelDescriptor.ActivationKind,
        layerTypes: [String]
    ) -> QWModelDescriptor {
        QWModelDescriptor(
            family: family,
            modelType: modelType,
            capabilities: [],
            hidden: 64,
            vocab: 64,
            denseIntermediate: 128,
            numLayers: layerTypes.count,
            layerTypes: layerTypes,
            fullAttentionInterval: 1,
            rmsEps: 1e-5,
            maxPositionEmbeddings: 256,
            qHeads: 2,
            kvHeads: 2,
            headDim: 32,
            attnOutputGate: false,
            partialRotaryFactor: 1.0,
            ropeTheta: 10_000.0,
            indexerHeads: nil,
            indexerKvHeads: nil,
            indexerHeadDim: nil,
            indexerBudget: nil,
            indexerCompressRatio: nil,
            numExperts: nil,
            expertsPerTok: nil,
            moeIntermediate: nil,
            sharedExpertIntermediate: nil,
            expertGateType: nil,
            hcCount: nil,
            hcLowrank: nil,
            gdnQKHeads: 0,
            gdnVHeads: 0,
            gdnHeadDim: 0,
            gdnQKVChannels: 0,
            gdnZChannels: 0,
            gdnConvKernel: 0,
            pleLayerIds: nil,
            pleEmbedDim: nil,
            pleConvKernel: nil,
            ngramSize: nil,
            headsPerNgram: nil,
            ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil,
            splitNgramParts: nil,
            mtpLayers: 0,
            mtpUseDedicatedEmbeddings: false,
            eosTokenId: 2,
            visDepth: 0,
            hasQKNorm: false,
            tieWordEmbeddings: true,
            activationKind: activationKind,
            normKind: normKind,
            useSandwichNorm: false,
            decisionConfig: QWDecisionModelConfig(framing: framing, topology: topology, head: head)
        )
    }

    private func writeSyntheticSafeTensorsPack(
        at dir: URL,
        tensors: [String: (shape: [Int], values: [Float])]
    ) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var header: [String: Any] = [:]
        var blob = Data()
        for name in tensors.keys.sorted() {
            let entry = tensors[name]!
            let start = blob.count
            for f in entry.values {
                var bits = f.bitPattern
                if (bits & 0x7F80_0000) != 0x7F80_0000 {
                    bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                }
                var bf = UInt16(truncatingIfNeeded: bits >> 16).littleEndian
                withUnsafeBytes(of: &bf) { blob.append(contentsOf: $0) }
            }
            header[name] = [
                "dtype": "BF16",
                "shape": entry.shape,
                "data_offsets": [start, blob.count],
            ]
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var paddedHeader = headerData
        while paddedHeader.count % 8 != 0 { paddedHeader.append(0x20) }
        var fileData = Data()
        var hdrLen = UInt64(paddedHeader.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { fileData.append(contentsOf: $0) }
        fileData.append(paddedHeader)
        fileData.append(blob)
        try fileData.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    private func deterministicWeights(count: Int, scale: Float = 0.05, seed: Int = 1) -> [Float] {
        (0..<count).map { i in
            let x = Float(((i + seed * 17) % 29) - 14) / 14.0
            return x * scale
        }
    }

    func testCanLoadAcceptsAllFourHeadsAndLayerNormAndGELU() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_canload_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try? writeSyntheticSafeTensorsPack(at: dir, tensors: ["model.embed_tokens.weight": ([64, 64], [Float](repeating: 0, count: 64 * 64))])
        let root = QWModelRoot(url: dir)

        let markerDesc = makeDescriptor(
            family: .lfm2Encoder, modelType: "lfm2_encoder",
            head: .markerScorer(hiddenSize: 64, headLayers: 2, chains: ["default"]),
            topology: .bidirectionalEncoder(isolateMediaPrefix: true),
            framing: .optionMarkers(layout: .stateQuestionOptions, markerToken: "<|mask|>", bosToken: "<bos>", maxLength: 256, imageTextLength: nil, audioTextLength: nil),
            normKind: .layerNorm(eps: 1e-5, hasBias: true),
            activationKind: .gelu,
            layerTypes: ["encoder_attention"]
        )
        XCTAssertTrue(QWDecisionPackEngine.canLoad(descriptor: markerDesc, root: root))

        let pooledDesc = makeDescriptor(
            family: .modernBert, modelType: "modernbert",
            head: .pooledClassifier(pooling: .meanNonPad, numLabels: 2, id2label: [0: "no", 1: "yes"]),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            framing: .sequencePair(maxLength: 256),
            normKind: .layerNorm(eps: 1e-5, hasBias: false),
            activationKind: .geluTanh,
            layerTypes: ["full_attention"]
        )
        XCTAssertTrue(QWDecisionPackEngine.canLoad(descriptor: pooledDesc, root: root))

        let jointDesc = makeDescriptor(
            family: .qwen3Dense, modelType: "qwen3",
            head: .jointOptionProjector(hiddenSize: 32),
            topology: .causal(supportsPrefixCache: true),
            framing: .turnVerbalizer(codesPerOption: 1, numbers: false, packSet: false, maxSetSize: 1, colonSlots: false, stateSeparator: "\n\n"),
            normKind: .rmsNorm(eps: 1e-6, hasUnitOffset: false),
            activationKind: .silu,
            layerTypes: ["full_attention"]
        )
        XCTAssertTrue(QWDecisionPackEngine.canLoad(descriptor: jointDesc, root: root))
    }

    func testMetalExecutionMarkerScorerPooledClassifierAndEmbeddings() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device unavailable")
        }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("qwg_gen_metal_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let h = 64
        let ffn = 128
        let ones = [Float](repeating: 1.0, count: h)
        let zeros = [Float](repeating: 0.0, count: h)
        let tensors: [String: (shape: [Int], values: [Float])] = [
            "model.embed_tokens.weight": ([64, h], deterministicWeights(count: 64 * h, scale: 0.25, seed: 3)),
            "model.norm.weight": ([h], ones),
            "model.norm.bias": ([h], zeros),
            "model.layers.0.input_layernorm.weight": ([h], ones),
            "model.layers.0.input_layernorm.bias": ([h], zeros),
            "model.layers.0.self_attn.q_proj.weight": ([h, h], deterministicWeights(count: h * h, seed: 5)),
            "model.layers.0.self_attn.k_proj.weight": ([h, h], deterministicWeights(count: h * h, seed: 7)),
            "model.layers.0.self_attn.v_proj.weight": ([h, h], deterministicWeights(count: h * h, seed: 9)),
            "model.layers.0.self_attn.o_proj.weight": ([h, h], deterministicWeights(count: h * h, seed: 11)),
            "model.layers.0.ffn_norm.weight": ([h], ones),
            "model.layers.0.ffn_norm.bias": ([h], zeros),
            "model.layers.0.mlp.fc1.weight": ([ffn, h], deterministicWeights(count: ffn * h, seed: 13)),
            "model.layers.0.mlp.fc1.bias": ([ffn], [Float](repeating: 0.01, count: ffn)),
            "model.layers.0.mlp.fc2.weight": ([h, ffn], deterministicWeights(count: h * ffn, seed: 17)),
            "model.layers.0.mlp.fc2.bias": ([h], zeros),
            "head.0.weight": ([32, h], deterministicWeights(count: 32 * h, seed: 19)),
            "head.0.bias": ([32], [Float](repeating: 0.0, count: 32)),
            "head.2.weight": ([1, 32], deterministicWeights(count: 32, scale: 0.2, seed: 23)),
            "head.2.bias": ([1], [0.05]),
            "classifier.weight": ([2, h], deterministicWeights(count: 2 * h, scale: 0.2, seed: 29)),
            "classifier.bias": ([2], [0.1, -0.1]),
        ]
        try writeSyntheticSafeTensorsPack(at: dir, tensors: tensors)

        let markerDesc = makeDescriptor(
            family: .lfm2Encoder, modelType: "lfm2_encoder",
            head: .markerScorer(hiddenSize: h, headLayers: 2, chains: ["default"]),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            framing: .optionMarkers(layout: .stateQuestionOptions, markerToken: "<|mask|>", bosToken: "<bos>", maxLength: 256, imageTextLength: nil, audioTextLength: nil),
            normKind: .layerNorm(eps: 1e-5, hasBias: true),
            activationKind: .gelu,
            layerTypes: ["encoder_attention"]
        )
        let markerEngine = try QWDecisionPackEngine(packPath: dir.path, descriptor: markerDesc, isSidecar: false)
        let markerOut = try await markerEngine.evaluate(QWDecisionBackendBatchRequest(
            sequences: [QWDecisionSequenceInput(
                tokenIds: [1, 2, 3, 4, 5],
                topology: .bidirectionalEncoder(isolateMediaPrefix: false),
                markerPositions: [2, 4]
            )],
            slots: [QWDecisionSlotTarget(sequenceIndex: 0, tokenPosition: 4)]
        ))
        let markers = try XCTUnwrap(markerOut.slotOutputs.first?.markerLogits)
        XCTAssertEqual(markers.count, 2)
        XCTAssertTrue(markers.allSatisfy(\.isFinite))

        let embRes = try await markerEngine.embedBatch(
            tokenBatches: [[1, 2, 3, 4], [5, 6, 7, 8]],
            pooling: .meanNonPad,
            normalize: true,
            truncateDimensions: 32
        )
        XCTAssertEqual(embRes.embeddings.count, 2)
        for vec in embRes.embeddings {
            XCTAssertEqual(vec.count, 32)
            let norm = sqrt(vec.reduce(0.0) { $0 + Double($1) * Double($1) })
            XCTAssertEqual(norm, 1.0, accuracy: 1e-3)
        }

        let pooledDesc = makeDescriptor(
            family: .modernBert, modelType: "modernbert",
            head: .pooledClassifier(pooling: .meanNonPad, numLabels: 2, id2label: [0: "no", 1: "yes"]),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            framing: .sequencePair(maxLength: 256),
            normKind: .layerNorm(eps: 1e-5, hasBias: true),
            activationKind: .gelu,
            layerTypes: ["full_attention"]
        )
        let pooledEngine = try QWDecisionPackEngine(packPath: dir.path, descriptor: pooledDesc, isSidecar: false)
        let pooledOut = try await pooledEngine.evaluate(QWDecisionBackendBatchRequest(
            sequences: [QWDecisionSequenceInput(tokenIds: [1, 2, 3, 4], topology: .bidirectionalEncoder(isolateMediaPrefix: false))],
            slots: [QWDecisionSlotTarget(sequenceIndex: 0, tokenPosition: 0)]
        ))
        let pooledLogits = try XCTUnwrap(pooledOut.slotOutputs.first?.pooledLogits)
        XCTAssertEqual(pooledLogits.count, 2)
        XCTAssertTrue(pooledLogits.allSatisfy(\.isFinite))
    }
}
