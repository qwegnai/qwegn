// QWLoadSafetyHardeningTests.swift — CPU regression tests for PlaneLayout, PlaneIO, and AQRepacker.
import Foundation
import XCTest
@testable import QWLoad
import QWQuant
import QWCore

private extension PlaneLayout {
    static func isRawDecisionPlaneName(_ name: String) -> Bool {
        let tokens = [
            ".conv.", ".short_conv.", "decision_head", "classifier",
            "score_head", "pointer_head", "marker_head", "type_emb", "type_embedding",
        ]
        return tokens.contains { name.contains($0) }
    }
}

final class QWLoadSafetyHardeningTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwload-sec-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    func testPlaneLayoutGuardsZeroGroupNegativeDimsAndOverflow() {
        let zeroGroup = QWQTensorEntry(shape: [4, 64], hq4Off: 0, hq: QWHQFormat(bits: 4, group: 0))
        XCTAssertNil(PlaneLayout.hq4.byteCount(zeroGroup))

        let badBits = QWQTensorEntry(shape: [4, 64], hq4Off: 0, hq: QWHQFormat(bits: 7, group: 32))
        XCTAssertNil(PlaneLayout.hq4.byteCount(badBits))

        let negDim = QWQTensorEntry(shape: [-4, 64], rawOff: 0)
        XCTAssertNil(PlaneLayout.raw.byteCount(negDim))

        let overflow = QWQTensorEntry(shape: [Int.max / 2, 8], rawOff: 0)
        XCTAssertNil(PlaneLayout.raw.byteCount(overflow))
    }

    func testPlaneIOZeroByteRawLoadAndLoadRawOverflowGuard() throws {
        let hdrObj: [String: Any] = [
            "provenance": [
                "model_id": QWModelDescriptor.qwen38_27b.family.rawValue,
                "geometry_digest": QWModelDescriptor.qwen38_27b.geometryDigestV3Hex,
            ],
            "tensors": [
                "empty_raw": ["shape": [0, 8], "raw_off": 12],
            ],
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: hdrObj)
        var out = Data(QWQuant.magic.utf8)
        out.append(contentsOf: withUnsafeBytes(of: UInt64(hdrData.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(Data(repeating: 0, count: 16))
        out.append(hdrData)
        let packURL = dir.appendingPathComponent("empty.qwp")
        try out.write(to: packURL)

        let pack = try QWQuant.Pack(url: packURL, descriptor: .qwen38_27b)
        let io = PlaneIO(pack: pack, policy: LoadPolicy(descriptor: .qwen38_27b, env: ["QW_PLANE_BUFFERED": "1"]))
        let emptyData = try io.load("empty_raw", layout: .raw)
        XCTAssertTrue(emptyData.isEmpty)

        let scratch = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 1)
        defer { scratch.deallocate() }
        XCTAssertThrowsError(try io.loadRaw("empty_raw", elemOffset: -1, elems: 2, into: scratch))
        XCTAssertThrowsError(try io.loadRaw("empty_raw", elemOffset: 0, elems: Int.max, into: scratch))
    }

    func testAQRepackerLayoutGuardsZeroColsWithoutDivideByZero() throws {
        let header: [String: Any] = [
            "w.weight": ["dtype": "U32", "shape": [2, 4], "data_offsets": [0, 32]],
            "w.scales": ["dtype": "F16", "shape": [2, 1], "data_offsets": [32, 36]],
            "w.biases": ["dtype": "F16", "shape": [2, 1], "data_offsets": [36, 40]],
        ]
        let hdrJSON = try JSONSerialization.data(withJSONObject: header)
        var file = Data()
        file.append(contentsOf: withUnsafeBytes(of: UInt64(hdrJSON.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        file.append(hdrJSON)
        file.append(Data(repeating: 0, count: 40))
        let url = dir.appendingPathComponent("model.safetensors")
        try file.write(to: url)
        let st = try QWSafeTensorsContainer(url: url)

        XCTAssertThrowsError(try QWAQRepacker.layout(container: st, name: "w.weight", rows: 2, cols: 0))
        XCTAssertThrowsError(try QWAQRepacker.layout(container: st, name: "w.weight", rows: 0, cols: 32))
    }

    func testNormGammaShiftLaw() {
        XCTAssertEqual(QWCheckpointTensorLoader.normGammaShift(normGammaIsPlusOne: true, isMLXCheckpoint: false), 0.0)
        XCTAssertEqual(QWCheckpointTensorLoader.normGammaShift(normGammaIsPlusOne: true, isMLXCheckpoint: true), 1.0)
        XCTAssertEqual(QWCheckpointTensorLoader.normGammaShift(normGammaIsPlusOne: false, isMLXCheckpoint: false), 1.0)
        XCTAssertEqual(QWCheckpointTensorLoader.normGammaShift(normGammaIsPlusOne: false, isMLXCheckpoint: true), 1.0)
    }

    func testPreserveBF16DecisionWeightsPolicyAndPlaneNames() {
        let decCfg = QWDecisionModelConfig.synthesizedVerbalizer(vocabSize: 32000)
        var smallCaps = QWModelDescriptor.qwen38_27b.capabilities
        smallCaps.insert(.decision)
        let smallDesc = QWModelDescriptor(
            family: QWModelDescriptor.qwen38_27b.family,
            modelType: "lfm3_vl",
            capabilities: smallCaps,
            hidden: 1024, vocab: 32000, denseIntermediate: 2816, numLayers: 16,
            layerTypes: Array(repeating: "full_attention", count: 16),
            fullAttentionInterval: 1, rmsEps: 1e-6, maxPositionEmbeddings: 4096,
            qHeads: 8, kvHeads: 4, headDim: 128, attnOutputGate: false,
            partialRotaryFactor: 1.0, ropeTheta: 10000,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: 2,
            decisionConfig: decCfg
        )
        let smallPolicy = LoadPolicy(descriptor: smallDesc, env: [:])
        XCTAssertTrue(smallPolicy.preserveBF16DecisionWeights)
        XCTAssertEqual(smallPolicy.preferredSafetensorsPrecision, "bf16")

        let forcedHQ4 = LoadPolicy(descriptor: smallDesc, env: ["QW_DECISION_FORCE_HQ4": "1"])
        XCTAssertFalse(forcedHQ4.preserveBF16DecisionWeights)
        XCTAssertEqual(forcedHQ4.preferredSafetensorsPrecision, "hq4_g32")

        let nonDecision = LoadPolicy(descriptor: .qwen38_27b, env: [:])
        XCTAssertFalse(nonDecision.preserveBF16DecisionWeights)
        XCTAssertEqual(nonDecision.preferredSafetensorsPrecision, "hq4_g32")

        XCTAssertTrue(PlaneLayout.isRawDecisionPlaneName("model.layers.0.short_conv.conv.weight"))
        XCTAssertTrue(PlaneLayout.isRawDecisionPlaneName("marker_head.mlp.0.weight"))
        XCTAssertTrue(PlaneLayout.isRawDecisionPlaneName("classifier.weight"))
        XCTAssertFalse(PlaneLayout.isRawDecisionPlaneName("model.layers.0.self_attn.q_proj.weight"))
    }

    func testPlaneLayoutPrecedenceAndSpanRejectsRaw() throws {
        let both = QWQTensorEntry(shape: [4, 64], hq4Off: 128, rawOff: 0)
        XCTAssertEqual(PlaneLayout.resolve(both), .raw)

        let hdrObj: [String: Any] = [
            "provenance": [
                "model_id": QWModelDescriptor.qwen38_27b.family.rawValue,
                "geometry_digest": QWModelDescriptor.qwen38_27b.geometryDigestV3Hex,
            ],
            "tensors": [
                "raw_and_hq4": ["shape": [4, 64], "raw_off": 0, "hq4_off": 128],
            ],
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: hdrObj)
        var out = Data(QWQuant.magic.utf8)
        out.append(contentsOf: withUnsafeBytes(of: UInt64(hdrData.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Array(raw)
        })
        out.append(Data(repeating: 0, count: 16))
        out.append(hdrData)
        let packURL = dir.appendingPathComponent("precedence.qwp")
        try out.write(to: packURL)

        let pack = try QWQuant.Pack(url: packURL, descriptor: .qwen38_27b)
        XCTAssertThrowsError(try QWCheckpointTensorLoader.span(pack, "raw_and_hq4"))
    }
}
