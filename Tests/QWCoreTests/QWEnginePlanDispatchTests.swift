import XCTest
import Foundation
@testable import QWCore

/// The engine plan carries the bench's winner map and asks it which kernel
/// to dispatch. The map is data: the plan resolves a name and never
/// inspects the map's contents.
final class QWEnginePlanDispatchTests: XCTestCase {
    /// A two-shape map: one 4-bit G32 shape at two bucket sizes.
    private func map(winner: String = "qwg_gemv_aq4_g32_v4") -> QWKernelDispatchTable {
        QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                              entries: [
                                QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4, group: 32,
                                                      winner: winner),
                                QWKernelDispatchEntry(nBucket: 4096, op: "gemv", bits: 4, group: 32,
                                                      winner: "qwg_gemv_q4g_dx_h16_t4"),
                              ])
    }

    private func plan(with table: QWKernelDispatchTable?) throws -> QWEnginePlan {
        try QWEnginePlan(geometry: .qwen38_27b, modelId: "stub",
                         geometryDigest: "stub", prefillChunkRows: 512,
                         kvStorageMode: "paged", kernelDispatch: table)
    }

    func testThePlanCarriesTheWinnerMap() throws {
        let table = map()
        let plan = try plan(with: table)
        XCTAssertEqual(plan.kernelDispatch, table)
    }

    func testThePlanResolvesAKernelNameFromTheMap() throws {
        let plan = try plan(with: map())
        XCTAssertEqual(plan.kernelName(nBucket: 4096, op: "gemv", bits: 4, group: 32),
                       "qwg_gemv_q4g_dx_h16_t4")
    }

    func testAPlanWithoutAWinnerMapResolvesToNothing() throws {
        let plan = try plan(with: nil)
        XCTAssertNil(plan.kernelDispatch)
        XCTAssertNil(plan.kernelName(nBucket: 4096, op: "gemv", bits: 4, group: 32))
    }

    func testAnUnmeasuredShapeResolvesToNothing() throws {
        let plan = try plan(with: map())
        XCTAssertNil(plan.kernelName(nBucket: 4096, op: "gemv", bits: 8, group: 64),
                     "an unmeasured width must not borrow another width's winner")
        XCTAssertNil(plan.kernelName(nBucket: 4096, op: "mlp_swiglu", bits: 4, group: 32),
                     "an unmeasured operation must not borrow another operation's winner")
    }

    func testTheWinnerMapChangesTheDigest() throws {
        let bare = try plan(with: nil)
        let mapped = try plan(with: map())
        XCTAssertNotEqual(bare.digest, mapped.digest,
                          "the digest must bind the map, which decides the kernel")
    }

    func testAPlanWithoutAWinnerMapKeepsItsDigest() throws {
        // Pinned so a plan without a map cannot silently change the digest
        // every existing pack and PSO cache is keyed by. The value was read
        // off the tree before the winner map was added, so it is the old one.
        let plan = try plan(with: nil)
        XCTAssertEqual(plan.digest, "138f51ca40d5db3d0d0963282a41557a92dfb386140d1a11aa4a01cc69dd38f3")
    }

    func testTwoMapsThatSelectDifferentKernelsHaveDifferentDigests() throws {
        let a = try plan(with: map(winner: "qwg_gemv_aq4_g32_v4"))
        let b = try plan(with: map(winner: "qwg_gemv_q4g_dx_h16_t4"))
        XCTAssertNotEqual(a.digest, b.digest,
                          "two maps with the same shape count but different winners must differ")
    }

    func testTwoMapsWithTheSameEntriesInADifferentOrderShareADigest() throws {
        let forward = QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                            entries: [
                                                QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4, group: 32,
                                                                      winner: "a"),
                                                QWKernelDispatchEntry(nBucket: 4096, op: "gemv", bits: 4, group: 32,
                                                                      winner: "b"),
                                            ])
        let reversed = QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                             entries: forward.entries.reversed())
        let a = try plan(with: forward)
        let b = try plan(with: reversed)
        XCTAssertEqual(a.digest, b.digest,
                       "the map is a set of shapes, so its order must not move the digest")
    }

    func testTheLookupDoesNotDependOnTheOrderOfAMalformedMap() {
        let a = QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4, group: 32, winner: "aaa")
        let b = QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4, group: 32, winner: "bbb")
        let forward = QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                            entries: [a, b])
        let reversed = QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                             entries: [b, a])
        XCTAssertEqual(forward.winner(nBucket: 64, op: "gemv", bits: 4, group: 32),
                       reversed.winner(nBucket: 64, op: "gemv", bits: 4, group: 32),
                       "two entries for one shape are a malformed map, but the answer "
                       + "must not depend on the order they arrived in")
    }

    func testThePlanRefusesAMapThisBuildDoesNotUnderstand() {
        let wrong = QWKernelDispatchTable(schema: "qwg-bench-kernels/1", entries: [])
        XCTAssertThrowsError(try plan(with: wrong)) { error in
            XCTAssertEqual(error as? QWEnginePlanError,
                           .unsupportedDispatchSchema("qwg-bench-kernels/1"))
        }
    }

    func testThePlanRefusesAMapHoldingTwoEntriesForOneShape() {
        let twin = QWKernelDispatchTable(schema: QWKernelDispatchTable.currentSchema,
                                          entries: [
                                              QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4,
                                                                    group: 32, winner: "a"),
                                              QWKernelDispatchEntry(nBucket: 64, op: "gemv", bits: 4,
                                                                    group: 32, winner: "b"),
                                          ])
        XCTAssertThrowsError(try plan(with: twin)) { error in
            XCTAssertEqual(error as? QWEnginePlanError,
                           .duplicateDispatchShape("64|gemv|4|32"))
        }
    }

    func testAPlanEncodedBeforeTheMapExistedStillDecodes() throws {
        let plan = try plan(with: nil)
        let data = try JSONEncoder().encode(plan)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(object["kernelDispatch"],
                     "a plan with no map must not write the key at all")
        let stripped = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(QWEnginePlan.self, from: stripped)
        XCTAssertNil(decoded.kernelDispatch)
        XCTAssertEqual(decoded.digest, plan.digest)
    }

    func testAPlanCarryingAMapThisBuildDoesNotUnderstandDoesNotDecode() throws {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(plan(with: nil))) as? [String: Any])
        object["kernelDispatch"] = ["schema": "qwg-bench-kernels/999", "entries": []]
        let data = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try JSONDecoder().decode(QWEnginePlan.self, from: data)) { error in
            // A nested decoder wraps a thrown error in a DecodingError.
            XCTAssertEqual(unwrap(error) as? QWKernelDispatchTable.Error,
                           .unsupportedSchema("qwg-bench-kernels/999"))
        }
    }

    /// The error under a `DecodingError`, or the error itself.
    private func unwrap(_ error: Error) -> Error {
        if case .dataCorrupted(let context) = error as? DecodingError,
           let underlying = context.underlyingError {
            return underlying
        }
        return error
    }

    func testDecodingAMapRefusesWhatThePlanRefuses() throws {        let wrong = Data(#"{"schema":"qwg-bench-kernels/1","entries":[]}"#.utf8)
        XCTAssertThrowsError(try QWKernelDispatchTable.decode(wrong)) { error in
            XCTAssertEqual(error as? QWKernelDispatchTable.Error,
                           .unsupportedSchema("qwg-bench-kernels/1"))
        }
        let twin = Data("""
            {"schema":"qwg-bench-kernels/2","entries":[
              {"n_bucket":64,"op":"gemv","bits":4,"group":32,"winner":"a"},
              {"n_bucket":64,"op":"gemv","bits":4,"group":32,"winner":"b"}]}
            """.utf8)
        XCTAssertThrowsError(try QWKernelDispatchTable.decode(twin)) { error in
            XCTAssertEqual(error as? QWKernelDispatchTable.Error,
                           .duplicateShape("64|gemv|4|32"))
        }
    }
}
