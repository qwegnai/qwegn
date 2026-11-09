import XCTest
import Foundation
@testable import QWCore

/// The published-config audit behind the support flip. The **engine's own**
/// decoder decides every published checkpoint, so the counts the plan and
/// the spec quote are the counts the decoder produces.
final class QWStandardPublishedConfigTests: XCTestCase {
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("Fixtures")

    /// One published checkpoint's verdict.
    struct Verdict {
        let id: String
        let family: String
        let accepted: Bool
        let reason: String
    }

    /// The recorded checkpoints, each put through the real decoder.
    func audit() throws -> [Verdict] {
        let raw = try Data(contentsOf: Self.fixtures
            .appendingPathComponent("qwen3_published_configs.json"))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var out: [Verdict] = []
        for (id, value) in root.sorted(by: { $0.key < $1.key }) {
            let obj = try XCTUnwrap(value as? [String: Any])
            let family = try XCTUnwrap(obj["model_type"] as? String)
            let data = try JSONSerialization.data(withJSONObject: obj)
            do {
                _ = try QWStandardConfigDecoder.decode(data)
                out.append(Verdict(id: id, family: family, accepted: true, reason: ""))
            } catch {
                out.append(Verdict(id: id, family: family, accepted: false,
                                   reason: String(describing: error)))
            }
        }
        return out
    }

    func testTheRecordedSetIsThePublishedOne() throws {
        let verdicts = try audit()
        XCTAssertEqual(verdicts.count, 94)
        XCTAssertEqual(verdicts.filter { $0.family == "qwen3" }.count, 67)
        XCTAssertEqual(verdicts.filter { $0.family == "qwen3_moe" }.count, 27)
    }

    func testTheSparseFamilyHasNoRefusedPublishedCheckpoint() throws {
        let verdicts = try audit().filter { $0.family == "qwen3_moe" }
        let refused = verdicts.filter { !$0.accepted }
        XCTAssertTrue(refused.isEmpty,
                      "the sparse family has refused published checkpoints: "
                      + refused.map { "\($0.id): \($0.reason)" }.joined(separator: ", "))
    }

    func testTheDenseFamilyRefusesOnlyItsTiedAndGuardCheckpoints() throws {
        let verdicts = try audit().filter { $0.family == "qwen3" }
        let refused = verdicts.filter { !$0.accepted }
        XCTAssertEqual(refused.count, 38)
        XCTAssertEqual(verdicts.filter { $0.accepted }.count, 29)
        // Every refusal names its feature: the tied head the engine cannot
        // materialise, or a guard checkpoint, which is not a causal LM.
        var tied = 0
        var guardModels = 0
        for v in refused {
            if v.reason.contains("tie_word_embeddings") { tied += 1 }
            if v.reason.contains("Qwen3ForGuardModel") { guardModels += 1 }
        }
        XCTAssertEqual(tied, 35, "the tied-head refusals")
        XCTAssertEqual(guardModels, 3, "the guard-architecture refusals")
        for v in refused {
            XCTAssertTrue(v.reason.contains("'"), "\(v.id): the refusal names nothing")
        }
    }

    func testEveryRefusalNamesItsFeature() throws {
        // A refusal that does not name the feature is a refusal a reader
        // cannot act on, so the reason must carry the name.
        for v in try audit() where !v.accepted {
            XCTAssertTrue(v.reason.contains("'"), "\(v.id): the refusal names nothing")
            XCTAssertFalse(v.reason.contains("config geometry"), "\(v.id): not a geometry fault")
        }
    }
}
