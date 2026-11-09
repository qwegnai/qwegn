import XCTest
import Foundation
@testable import QWFlashNext

// Goldens for the recurrent blocks: GatedDeltaNet and the PLE layer (HF qwen4_exp, Scripts/flashnext/gen_goldens.py).
final class FNRecurrentGoldenTests: XCTestCase {
    func testGatedDeltaNetMatchesHF() throws {
        let g = try FNGoldenTests.load()["gated_delta_net"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        func t(_ key: String) -> [Float] { FNGoldenTests.tensor(w[key]) }
        let gdn = FNGatedDeltaNet(
            hidden: g["hidden"] as! Int, kHeads: g["k_heads"] as! Int, vHeads: g["v_heads"] as! Int,
            kDim: g["k_dim"] as! Int, vDim: g["v_dim"] as! Int, convKernel: g["conv_kernel"] as! Int,
            eps: Float(g["eps"] as! Double),
            inProjQKV: t("in_proj_qkv.weight"), inProjZ: t("in_proj_z.weight"),
            inProjB: t("in_proj_b.weight"), inProjA: t("in_proj_a.weight"),
            convWeight: t("conv1d.weight"), dtBias: t("dt_bias"), aLog: t("A_log"),
            normWeight: t("norm.weight"), outProj: t("out_proj.weight"))
        // forward() is one sequence: run each batch row independently (conv/state never cross rows).
        let input = FNGoldenTests.tensor(g["input"])
        let hidden = g["hidden"] as! Int
        let seqLen = ((g["input_shape"] as? [Int]) ?? [2, 5, hidden])[1]
        let perSeq = seqLen * hidden
        var out: [Float] = []
        for b in 0..<(input.count / perSeq) {
            out += gdn.forward(Array(input[(b * perSeq)..<((b + 1) * perSeq)]), rows: seqLen)
        }
        XCTAssertLessThan(FNGoldenTests.maxAbsError(out, FNGoldenTests.tensor(g["output"])), 1e-4)
    }

    func testPLELayerMatchesHF() throws {
        let g = try FNGoldenTests.load()["ple_layer"] as! [String: Any]
        let w = g["weights"] as! [String: Any]
        func t(_ key: String) -> [Float] { FNGoldenTests.tensor(w[key]) }
        let hidden = g["hidden"] as! Int, hc = g["hc"] as! Int, embedDim = g["ple_embed_dim"] as! Int
        let ple = FNPLE(hidden: hidden, hc: hc, embedDim: embedDim, convKernel: g["conv_kernel"] as! Int,
                        dilation: g["dilation"] as! Int, eps: Float(g["eps"] as! Double),
                        keyProj: t("key_proj.weight"), valueProj: t("value_proj.weight"),
                        normKey: t("norm_key.weight"), normQuery: t("norm_query.weight"),
                        normConv: t("norm_conv.weight"), convWeight: t("conv1d.weight"))
        // N-gram lookup via the verified FNNgram ids and the HF embedding table.
        let table = FNGoldenTests.tensor(g["ngram_table"])
        let tableShape = (g["ngram_table"] as! [String: Any])["shape"] as! [Int]
        let headDim = tableShape[1]
        let seqs = g["ids"] as! [[Int]]
        let seqLen = seqs[0].count
        var embedding: [Float] = []
        for seq in seqs {
            let rowsIds = FNNgram.ids(tokens: seq.map { Int64($0) }, vocab: g["vocab"] as! Int,
                                      ngram: g["ngram_size"] as! Int, headsPerNgram: g["heads_per_ngram"] as! Int,
                                      baseVocab: g["ngram_vocab_size_base"] as! Int, seed: g["seed"] as! Int,
                                      pleLayerIndex: g["ple_layer_index"] as! Int, eos: Int64(g["eos"] as! Int))
            for row in rowsIds {
                for id in row { embedding.append(contentsOf: table[(Int(id) * headDim)..<((Int(id) + 1) * headDim)]) }
            }
        }
        XCTAssertLessThan(FNGoldenTests.maxAbsError(embedding, FNGoldenTests.tensor(g["embedding"])), 1e-6)
        let h = FNGoldenTests.tensor(g["hidden_input"])
        var out: [Float] = []
        for b in 0..<seqs.count {
            let eb = Array(embedding[(b * seqLen * embedDim)..<((b + 1) * seqLen * embedDim)])
            let hb = Array(h[(b * seqLen * hidden * hc)..<((b + 1) * seqLen * hidden * hc)])
            out += ple.forward(embedding: eb, hiddenState: hb, rows: seqLen)
        }
        XCTAssertLessThan(FNGoldenTests.maxAbsError(out, FNGoldenTests.tensor(g["output"])), 1e-4)
    }
}
