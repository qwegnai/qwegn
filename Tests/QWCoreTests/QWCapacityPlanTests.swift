import XCTest
@testable import QWCore

final class QWCapacityPlanTests: XCTestCase {
    private let GiB = 1 << 30
    private let native = 262_144

    func testDefaultsToNativeWindowWhenNothingRequested() throws {
        let plan = try QWCapacityPlan.plan(nativeWindow: native, requested: nil, perTokenBytes: 65_536,
                                           fixedBytes: 6 * GiB, availableBytes: 100 * GiB, yarnAllowed: false)
        XCTAssertEqual(plan.context, native)
        XCTAssertNil(plan.yarnFactor)
        XCTAssertTrue(plan.fits)
        XCTAssertEqual(plan.requiredBytes, 6 * GiB + 65_536 * native)
    }

    func testRequestBelowNativeIsHonouredExactly() throws {
        let plan = try QWCapacityPlan.plan(nativeWindow: native, requested: 8_192, perTokenBytes: 65_536,
                                           fixedBytes: GiB, availableBytes: 100 * GiB, yarnAllowed: false)
        XCTAssertEqual(plan.context, 8_192)
        XCTAssertNil(plan.yarnFactor)
    }

    func testLongContextUsesYarnFactorWhenAllowed() throws {
        let plan = try QWCapacityPlan.plan(nativeWindow: native, requested: 524_288, perTokenBytes: 65_536,
                                           fixedBytes: 6 * GiB, availableBytes: 100 * GiB, yarnAllowed: true)
        XCTAssertEqual(plan.context, 524_288)
        XCTAssertEqual(plan.yarnFactor ?? 0, 2.0, accuracy: 1e-12)
        XCTAssertTrue(plan.fits)
    }

    func testOneMillionWithYarnGivesFactorFour() throws {
        let plan = try QWCapacityPlan.plan(nativeWindow: native, requested: 1_048_576, perTokenBytes: 16_384,
                                           fixedBytes: 6 * GiB, availableBytes: 100 * GiB, yarnAllowed: true)
        XCTAssertEqual(plan.yarnFactor ?? 0, 4.0, accuracy: 1e-12)
    }

    func testBeyondNativeWithoutYarnFailsLoudly() {
        XCTAssertThrowsError(try QWCapacityPlan.plan(nativeWindow: native, requested: 1_048_576,
                                                     perTokenBytes: 65_536, fixedBytes: GiB,
                                                     availableBytes: 100 * GiB, yarnAllowed: false)) { error in
            XCTAssertEqual(error as? QWCapacityPlan.Failure,
                           .yarnNotAvailable(requested: 1_048_576, native: native))
        }
    }

    func testNonPositiveContextIsRejected() {
        XCTAssertThrowsError(try QWCapacityPlan.plan(nativeWindow: native, requested: 0, perTokenBytes: 1,
                                                     fixedBytes: 0, availableBytes: GiB, yarnAllowed: false)) { error in
            XCTAssertEqual(error as? QWCapacityPlan.Failure, .contextBelowMinimum(0))
        }
    }

    func testDoesNotFitReportsLargestFittingContext() throws {
        // 10 GiB available, 6 GiB fixed, 64 KiB per token → 4 GiB / 64 KiB = 65,536 tokens.
        let plan = try QWCapacityPlan.plan(nativeWindow: native, requested: native, perTokenBytes: 65_536,
                                           fixedBytes: 6 * GiB, availableBytes: 10 * GiB, yarnAllowed: false)
        XCTAssertFalse(plan.fits)
        XCTAssertEqual(plan.maxFittingContext, 65_536)
    }

    func testTwentySevenBKVBytesPerTokenByMode() {
        // Qwen3.8-27B: 16 attention layers, 4 KV heads, head dim 256.
        XCTAssertEqual(QWCapacityPlan.kvBytesPerToken(attentionLayers: 16, kvHeads: 4, headDim: 256,
                                                      bytesPerElement: 2.0), 65_536)
        XCTAssertEqual(QWCapacityPlan.kvBytesPerToken(attentionLayers: 16, kvHeads: 4, headDim: 256,
                                                      bytesPerElement: 1.0), 32_768)
        XCTAssertEqual(QWCapacityPlan.kvBytesPerToken(attentionLayers: 16, kvHeads: 4, headDim: 256,
                                                      bytesPerElement: 0.5), 16_384)
    }

    func testSteppedCapacityAndFormatLogLine() {
        XCTAssertEqual(QWCapacityPlan.steppedCapacity(4096), 4096)
        XCTAssertEqual(QWCapacityPlan.steppedCapacity(4100), 4128)
        XCTAssertEqual(QWCapacityPlan.steppedCapacity(8192), 8192)
        XCTAssertEqual(QWCapacityPlan.steppedCapacity(8193), 16384)
        XCTAssertEqual(QWCapacityPlan.steppedCapacity(262_144), 262_144)

        let log = QWCapacityPlan.formatLogLine(
            weightsBytes: 2 * GiB,
            fixedScratchBytes: GiB,
            kvMode: "f16",
            perTokenBytes: 31_296,
            maxTokens: 8192,
            totalPlannedBytes: 3 * GiB + 8192 * 31_296,
            availableBytes: 40 * GiB
        )
        XCTAssertTrue(log.contains("[memory] weights=2.00 GiB fixed_scratch=1.00 GiB kv_mode=f16 kv_per_tok=31296 B max_tokens=8192"))
    }
}
