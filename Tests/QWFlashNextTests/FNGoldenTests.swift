import XCTest
import Foundation
@testable import QWFlashNext

// Goldens: Scripts/flashnext/gen_goldens.py (HF qwen4_exp, seed 20261020).
final class FNGoldenTests: XCTestCase {
    static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("fixtures/fn_goldens.json")

    static func load() throws -> [String: Any] {
        let data = try Data(contentsOf: fixtureURL)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    static func tensor(_ obj: Any?) -> [Float] {
        let dict = obj as! [String: Any]
        return (dict["data"] as! [Double]).map { Float($0) }
    }

    static func maxAbsError(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        var worst: Float = 0
        for (x, y) in zip(a, b) { worst = max(worst, abs(x - y)) }
        return worst
    }

    func testNgramConstantsMatchHF() throws {
        let g = try Self.load()["ngram_ids"] as! [String: Any]
        let mult = (g["multipliers"] as! [Int]).map { Int64($0) }
        XCTAssertEqual(FNNgram.layerMultipliers(vocab: g["vocab"] as! Int, ngram: g["ngram_size"] as! Int,
                                                pleLayerIndex: g["ple_layer_index"] as! Int,
                                                seed: g["seed"] as! Int), mult)
        let table = FNNgram.headTable(baseVocab: g["ngram_vocab_size_base"] as! Int,
                                      headCount: (g["ngram_size"] as! Int - 1) * (g["heads_per_ngram"] as! Int),
                                      pleLayerIndex: g["ple_layer_index"] as! Int)
        XCTAssertEqual(table.sizes, g["head_sizes"] as! [Int])
        XCTAssertEqual(table.offsets, g["head_offsets"] as! [Int])
        XCTAssertEqual(table.totalRows, g["total_rows"] as! Int)
    }

    func testNgramIdsMatchHFExactly() throws {
        let g = try Self.load()["ngram_ids"] as! [String: Any]
        for case_ in g["cases"] as! [[String: Any]] {
            let tokens = (case_["tokens"] as! [Int]).map { Int64($0) }
            let want = case_["ids"] as! [[Int]]
            let got = FNNgram.ids(tokens: tokens, vocab: g["vocab"] as! Int, ngram: g["ngram_size"] as! Int,
                                  headsPerNgram: g["heads_per_ngram"] as! Int,
                                  baseVocab: g["ngram_vocab_size_base"] as! Int, seed: g["seed"] as! Int,
                                  pleLayerIndex: g["ple_layer_index"] as! Int, eos: Int64(g["eos"] as! Int))
            XCTAssertEqual(got.map { $0.map { Int($0) } }, want)
        }
    }

    func testRMSNormMatchesHF() throws {
        let g = try Self.load()["rmsnorm"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        let input = Self.tensor(g["input"])
        let out = FNMath.groupRMSNorm(input, rows: input.count / 64, width: 64,
                                      group: g["group"] as! Int,
                                      weight: Self.tensor(w["weight"]), eps: Float(g["eps"] as! Double))
        XCTAssertLessThan(Self.maxAbsError(out, Self.tensor(g["output"])), 1e-5)
    }

    func testHyperConnectionMatchesHF() throws {
        let g = try Self.load()["hc_gated_residual"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        let hc = FNHyperConnection(hc: g["hc"] as! Int, hidden: g["hidden"] as! Int, lowrank: g["lowrank"] as! Int,
                                   eps: Float(g["eps"] as! Double),
                                   normWeight: Self.tensor(w["hc_norm.weight"]),
                                   downWeight: Self.tensor(w["input_mix_weight_down.weight"]),
                                   upWeight: Self.tensor(w["input_mix_weight_up.weight"]),
                                   injectWeight: Self.tensor(w["block_inject_weight.weight"]))
        let input = Self.tensor(g["input"])
        let width = (g["hc"] as! Int) * (g["hidden"] as! Int)
        let rows = input.count / width
        let (mixed, inj) = hc.combine(input, rows: rows)
        XCTAssertLessThan(Self.maxAbsError(mixed, Self.tensor(g["mixed"])), 1e-5)
        XCTAssertLessThan(Self.maxAbsError(inj, Self.tensor(g["injection"])), 1e-5)
    }

    func testSparseMoEMatchesHF() throws {
        let g = try Self.load()["sparse_moe"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        let h = g["hidden"] as! Int
        let moe = FNSparseMoE(hidden: h, experts: g["experts"] as! Int, topK: g["top_k"] as! Int,
                              intermediate: g["expert_intermediate"] as! Int,
                              sharedIntermediate: g["shared_intermediate"] as! Int,
                              normTopK: g["norm_topk_prob"] as! Bool,
                              gate: Self.tensor(w["gate.weight"]),
                              source: FNArrayExperts(experts: g["experts"] as! Int,
                                                     gateUp: Self.tensor(w["experts.gate_up_proj"]),
                                                     down: Self.tensor(w["experts.down_proj"])),
                              sharedGate: Self.tensor(w["shared_expert.gate_proj.weight"]),
                              sharedUp: Self.tensor(w["shared_expert.up_proj.weight"]),
                              sharedDown: Self.tensor(w["shared_expert.down_proj.weight"]),
                              sharedGateScalar: Self.tensor(w["shared_expert_gate.weight"]))
        let input = Self.tensor(g["input"])
        let out = moe.forward(input, rows: input.count / h)
        XCTAssertLessThan(Self.maxAbsError(out, Self.tensor(g["output"])), 1e-4)
    }
}
