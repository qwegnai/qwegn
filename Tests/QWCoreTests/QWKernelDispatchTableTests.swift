import XCTest
import Foundation
@testable import QWCore

final class QWKernelDispatchTableTests: XCTestCase {
    private let json = """
    {
      "schema": "qwg-bench-kernels/2",
      "entries": [
        {"n_bucket": 64, "op": "gemv", "bits": 4, "group": 32,
         "winner": "qwg_gemv_aq4_g32_v4", "runner_up": "qwg_gemv_q4g_v4", "margin": 0.017},
        {"n_bucket": 2048, "op": "gemv", "bits": 4, "group": 32,
         "winner": "qwg_gemv_q4g_v4", "runner_up": "qwg_gemv_aq4_g32_v4", "margin": 0.004}
      ]
    }
    """

    func testDecodesTheWinnerMap() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        XCTAssertEqual(table.entries.count, 2)
        XCTAssertEqual(table.schema, "qwg-bench-kernels/2")
        XCTAssertEqual(table.entries[0].winner, "qwg_gemv_aq4_g32_v4")
        XCTAssertEqual(table.entries[0].nBucket, 64)
        XCTAssertEqual(table.entries[0].bits, 4)
        XCTAssertEqual(table.entries[0].group, 32)
    }

    func testTheMapRoundTripsThroughItsEncoder() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        let again = try QWKernelDispatchTable.decode(table.encoded())
        XCTAssertEqual(again.entries, table.entries)
    }

    func testAnUnknownSchemaIsRejected() {
        let bad = """
        {"schema": "something-else/9", "entries": []}
        """
        XCTAssertThrowsError(try QWKernelDispatchTable.decode(Data(bad.utf8))) { error in
            XCTAssertEqual(error as? QWKernelDispatchTable.Error, .unsupportedSchema("something-else/9"))
        }
    }

    func testTwoEntriesForOneShapeAreRejected() {
        let dup = """
        {"schema": "qwg-bench-kernels/2", "entries": [
          {"n_bucket": 64, "op": "gemv", "bits": 4, "group": 32, "winner": "a"},
          {"n_bucket": 64, "op": "gemv", "bits": 4, "group": 32, "winner": "b"}
        ]}
        """
        XCTAssertThrowsError(try QWKernelDispatchTable.decode(Data(dup.utf8))) { error in
            XCTAssertEqual(error as? QWKernelDispatchTable.Error, .duplicateShape("64|gemv|4|32"))
        }
    }

    func testTheWinnerIsTheLargestBucketAtOrBelowTheShape() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        XCTAssertEqual(table.winner(nBucket: 64, op: "gemv", bits: 4, group: 32),
                       "qwg_gemv_aq4_g32_v4")
        XCTAssertEqual(table.winner(nBucket: 2048, op: "gemv", bits: 4, group: 32),
                       "qwg_gemv_q4g_v4")
        XCTAssertEqual(table.winner(nBucket: 1024, op: "gemv", bits: 4, group: 32),
                       "qwg_gemv_aq4_g32_v4")
    }

    func testAShapeBelowEveryMeasuredBucketHasNoWinner() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        XCTAssertNil(table.winner(nBucket: 32, op: "gemv", bits: 4, group: 32),
                     "the map must not answer with a candidate measured at a larger shape")
    }

    func testAnUnmeasuredShapeHasNoWinner() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        XCTAssertNil(table.winner(nBucket: 2048, op: "gemv", bits: 8, group: 64))
        XCTAssertNil(table.winner(nBucket: 2048, op: "gemv_incumbent", bits: 4, group: 32))
    }

    func testTheRunnerUpAndItsMarginAreCarried() throws {
        let table = try QWKernelDispatchTable.decode(Data(json.utf8))
        XCTAssertEqual(table.entries[0].runnerUp, "qwg_gemv_q4g_v4")
        XCTAssertEqual(table.entries[0].margin ?? 0, 0.017, accuracy: 1e-9)
        XCTAssertEqual(table.entries[1].runnerUp, "qwg_gemv_aq4_g32_v4")
    }

    func testAnEntryWithoutARunnerUpDecodesWithNoMargin() throws {
        let solo = """
        {"schema": "qwg-bench-kernels/2", "entries": [
          {"n_bucket": 64, "op": "gemv", "bits": 4, "group": 32, "winner": "only"}
        ]}
        """
        let table = try QWKernelDispatchTable.decode(Data(solo.utf8))
        XCTAssertNil(table.entries[0].runnerUp)
        XCTAssertNil(table.entries[0].margin)
    }

    /// Two entries that differ only in the operation are two winners, not one
    /// shape twice: the op is part of the key, not a label on it.
    func testTwoEntriesThatDifferOnlyInOpAreTwoWinners() throws {
        let twoOps = """
        {"schema": "qwg-bench-kernels/2", "entries": [
          {"n_bucket": 14336, "op": "attn_qkv", "bits": 4, "group": 32,
           "winner": "qwg_gemv_q4g_dx_h16_t4"},
          {"n_bucket": 14336, "op": "mlp_swiglu", "bits": 4, "group": 32,
           "winner": "qwg_gemv_q4g_silu_pair_dx_h16"}
        ]}
        """
        let table = try QWKernelDispatchTable.decode(Data(twoOps.utf8))
        XCTAssertEqual(table.entries.count, 2)
        XCTAssertEqual(table.winner(nBucket: 14336, op: "attn_qkv", bits: 4, group: 32),
                       "qwg_gemv_q4g_dx_h16_t4")
        XCTAssertEqual(table.winner(nBucket: 14336, op: "mlp_swiglu", bits: 4, group: 32),
                       "qwg_gemv_q4g_silu_pair_dx_h16")
        XCTAssertNil(table.winner(nBucket: 14336, op: "lm_head", bits: 4, group: 32))
    }

    /// The op axis is only meaningful if the bench and the engine share the
    /// vocabulary, so the list lives in the type both of them use.
    func testTheKnownOpsCoverTheFamilySweepAndTheOperatorSuites() {
        for op in ["gemv", "mlx_nax_gemm", "gdn_inproj_decode", "mlp_swiglu",
                   "attn_qkv", "lm_head", "moe_gateup_decode",
                   "moe_gateup_prefill"] {
            XCTAssertTrue(QWKernelDispatchTable.knownOps.contains(op),
                          "the bench emits \(op), so it must be a known op")
        }
    }

    /// Reading a map never refuses an operation the reader has not heard of:
    /// a future operator must not be turned away by an older reader.
    func testAnUnknownOpIsStillReadBack() throws {
        let future = """
        {"schema": "qwg-bench-kernels/2", "entries": [
          {"n_bucket": 64, "op": "some_future_op", "bits": 4, "group": 32, "winner": "a"}
        ]}
        """
        let table = try QWKernelDispatchTable.decode(Data(future.utf8))
        XCTAssertEqual(table.entries.count, 1)
        XCTAssertEqual(table.winner(nBucket: 64, op: "some_future_op", bits: 4, group: 32), "a")
    }
}
