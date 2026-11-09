import XCTest
import QWCore
@testable import QWEngine

final class QWMTPAdaptiveDepthTests: XCTestCase {
    func testNormalizedEntropyIsBoundedAndRanksUncertainty() throws {
        let uniform: [Float] = [0, 0, 0, 0]
        let confident: [Float] = [20, 0, -5, -10]
        let uniformEntropy = try uniform.withUnsafeBufferPointer(QWMTPAdaptiveDepth.normalizedEntropy)
        let confidentEntropy = try confident.withUnsafeBufferPointer(QWMTPAdaptiveDepth.normalizedEntropy)
        XCTAssertEqual(uniformEntropy, 1, accuracy: 1e-12)
        XCTAssertGreaterThanOrEqual(confidentEntropy, 0)
        XCTAssertLessThan(confidentEntropy, uniformEntropy)
    }

    func testAdaptiveStopOnlyStopsAfterAProposalAndAtThreshold() throws {
        XCTAssertFalse(try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: 1, threshold: 0.5,
                                                         completedDrafts: 1, maximumDrafts: 1))
        XCTAssertTrue(try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: 0.5, threshold: 0.5,
                                                        completedDrafts: 1, maximumDrafts: 2))
        XCTAssertFalse(try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: 0.49, threshold: 0.5,
                                                         completedDrafts: 1, maximumDrafts: 2))
        XCTAssertFalse(try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: 1, threshold: 0.5,
                                                         completedDrafts: 2, maximumDrafts: 2))
        XCTAssertThrowsError(try QWMTPAdaptiveDepth.shouldStop(normalizedEntropy: .nan, threshold: 0.5,
                                                               completedDrafts: 1, maximumDrafts: 2))
    }

    func testEntropyThresholdDefaultsAndRejectsInvalidValues() throws {
        XCTAssertEqual(try QWEngineDriver.resolveMTPAdaptiveEntropyThreshold(nil), 0.5)
        XCTAssertEqual(try QWEngineDriver.resolveMTPAdaptiveEntropyThreshold("0.25"), 0.25)
        for invalid in ["", "nan", "inf", "-0.1", "1.1", "nope"] {
            XCTAssertThrowsError(try QWEngineDriver.resolveMTPAdaptiveEntropyThreshold(invalid))
        }
    }

    func testOptimalDraftCountIncreasesWithAcceptRateAndExpertOverlap() throws {
        let config: [String: Any] = [
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 5120,
            "moe_intermediate_size": 1280,
            "num_experts": 288,
            "num_experts_per_tok": 8,
            "norm_topk_prob": true,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "eos_token_id": 2,
            "vocab_size": 151552,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        let geom = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))

        let lowAccept = QWMTPAdaptiveDepth.optimalDraftCount(
            geometry: geom, trunkBits: 4, expertBits: 4, acceptRate: 0.15, maxDrafts: 3)
        let highAccept = QWMTPAdaptiveDepth.optimalDraftCount(
            geometry: geom, trunkBits: 4, expertBits: 4, acceptRate: 0.90, maxDrafts: 3)
        let coldStreamingGeom = geom.withResidentExperts(200)
        let coldStreamingDepth = QWMTPAdaptiveDepth.optimalDraftCount(
            geometry: coldStreamingGeom, trunkBits: 4, expertBits: 4, acceptRate: 0.80, maxDrafts: 3)
        let residentDepth = QWMTPAdaptiveDepth.optimalDraftCount(
            geometry: geom, trunkBits: 4, expertBits: 4, acceptRate: 0.80, maxDrafts: 3)
        XCTAssertEqual(lowAccept, 1)
        XCTAssertGreaterThan(highAccept, lowAccept)
        XCTAssertLessThanOrEqual(highAccept, 3)
        XCTAssertLessThanOrEqual(coldStreamingDepth, residentDepth)
    }

    func testMoESpecDraftExpertBudgetPreservesRootAndBoundsUnion() throws {
        let geom = try QWModelGeometry.stubPureMoE()
        let kTop = geom.expertsPerTok

        let budget1 = QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(geometry: geom, verifyRows: 1, acceptRate: 0.7)
        let budget3 = QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(geometry: geom, verifyRows: 3, acceptRate: 0.7)
        let explicit = QWMTPAdaptiveDepth.moeSpecDraftExpertBudget(geometry: geom, verifyRows: 3, explicitBudget: kTop + 4)
        XCTAssertEqual(budget1, kTop)
        XCTAssertGreaterThan(budget3, kTop)
        XCTAssertLessThan(budget3, 3 * kTop)
        XCTAssertEqual(explicit, kTop + 4)
    }
}
