import XCTest
import Foundation
@testable import QWFlashNext

// Goldens for full attention + QSA indexer (HF qwen4_exp, eager path, float causal mask).
final class FNAttentionGoldenTests: XCTestCase {
    static func attentionBlock() throws -> (FNAttention, [String: Any]) {
        let g = try FNGoldenTests.load()["attention"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        func t(_ key: String) -> [Float] { FNGoldenTests.tensor(w[key]) }
        let ix = g["indexer"] as! [String: Any]
        let indexer = FNQSAIndexer(heads: ix["heads"] as! Int, kvHeads: ix["kv_heads"] as! Int,
                                   headDim: ix["head_dim"] as! Int, budget: ix["budget"] as! Int,
                                   compress: ix["compress"] as! Int, eps: Float(ix["eps"] as! Double),
                                   qkProj: t("indexer.index_qk_proj.weight"),
                                   qNorm: t("indexer.q_layernorm.weight"), kNorm: t("indexer.k_layernorm.weight"))
        let block = FNAttention(
            hidden: g["hidden"] as! Int, heads: g["heads"] as! Int, kvHeads: g["kv_heads"] as! Int,
            headDim: g["head_dim"] as! Int, eps: Float(g["eps"] as! Double),
            qProj: t("q_proj.weight"), kProj: t("k_proj.weight"), vProj: t("v_proj.weight"),
            oProj: t("o_proj.weight"), qNorm: t("q_norm.weight"), kNorm: t("k_norm.weight"),
            indexer: indexer,
            rope: FNRope(rotaryDim: g["rotary_dim"] as! Int, theta: g["rope_theta"] as! Double))
        return (block, g)
    }

    func testRopeTableMatchesHF() throws {
        let (block, g) = try Self.attentionBlock()
        let positions = Array(0..<20)
        let (cos, sin) = block.rope.tables(positions: positions)
        XCTAssertLessThan(FNGoldenTests.maxAbsError(cos, FNGoldenTests.tensor(g["rope_cos"])), 1e-5)
        XCTAssertLessThan(FNGoldenTests.maxAbsError(sin, FNGoldenTests.tensor(g["rope_sin"])), 1e-5)
    }

    func testIndexerSelectionMatchesHFExactly() throws {
        let (block, g) = try Self.attentionBlock()
        let input = FNGoldenTests.tensor(g["input"])
        let hidden = g["hidden"] as! Int
        let want = g["selected"] as! [[[Int]]]
        for (b, batch) in want.enumerated() {
            let L = batch.count
            let x = Array(input[(b * L * hidden)..<((b + 1) * L * hidden)])
            let sel = block.indexer.select(x, rows: L, hidden: hidden, rope: block.rope)
            for t in 0..<L { for j in 0..<L { XCTAssertEqual(sel[t][j], batch[t][j] == 1, "b\(b) t\(t) j\(j)") } }
        }
    }

    func testAttentionMatchesHF() throws {
        let (block, g) = try Self.attentionBlock()
        let input = FNGoldenTests.tensor(g["input"])
        let hidden = g["hidden"] as! Int
        let L = (g["selected"] as! [[[Int]]])[0].count
        var out: [Float] = []
        for b in 0..<(input.count / (L * hidden)) {
            out += block.forward(Array(input[(b * L * hidden)..<((b + 1) * L * hidden)]), rows: L)
        }
        XCTAssertLessThan(FNGoldenTests.maxAbsError(out, FNGoldenTests.tensor(g["output"])), 1e-4)
    }

    func testRopeYaRN1MScaling() {
        let defaultRope = FNRope(rotaryDim: 64, theta: 10_000_000.0)
        let yarnRope = FNRope(rotaryDim: 64, theta: 10_000_000.0,
                              yarnFactor: 4.0, originalMaxPosition: 262_144)
        XCTAssertEqual(defaultRope.attentionScaling, 1.0, accuracy: 1e-6)
        let expectedScale = Float(0.1 * Foundation.log(4.0) + 1.0)
        XCTAssertEqual(yarnRope.attentionScaling, expectedScale, accuracy: 1e-6)

        let (c0, s0) = yarnRope.tables(positions: [0, 524_288, 1_048_575])
        XCTAssertEqual(c0[0], expectedScale, accuracy: 1e-6)
        XCTAssertEqual(s0[0], 0.0, accuracy: 1e-6)
        XCTAssertTrue(c0.allSatisfy { $0.isFinite && abs($0) <= expectedScale + 1e-5 })
        XCTAssertTrue(s0.allSatisfy { $0.isFinite && abs($0) <= expectedScale + 1e-5 })

        let (cStride, sStride) = yarnRope.tables(count: 2, stride: 524_288)
        for i in 0..<64 {
            XCTAssertEqual(cStride[64 + i], c0[64 + i], accuracy: 1e-6)
            XCTAssertEqual(sStride[64 + i], s0[64 + i], accuracy: 1e-6)
        }
    }
}
