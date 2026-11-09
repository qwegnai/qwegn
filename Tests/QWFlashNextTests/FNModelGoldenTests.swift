import XCTest
import Foundation
@testable import QWFlashNext

// Full-model golden: 4 layers (GDN, full attention, GDN+PLE, full attention), HF Qwen4ExpForCausalLM at small dims.
final class FNModelGoldenTests: XCTestCase {
    struct Weights {
        let map: [String: [String: Any]]
        func data(_ key: String) -> [Float] { FNGoldenTests.tensor(map[key]) }
        func shape(_ key: String) -> [Int] { map[key]!["shape"] as! [Int] }
    }

    static func buildModel() throws -> (FNModel, [String: Any]) {
        let g = try FNGoldenTests.load()["model"] as! [String: Any]
        let w = Weights(map: g["weights"] as! [String: [String: Any]])
        let hidden = g["hidden"] as! Int, hc = g["hc"] as! Int, vocab = g["vocab"] as! Int
        let eps: Float = 1e-6   // Qwen4ExpConfig rms_norm_eps in gen_goldens.model_config
        let types = g["layer_types"] as! [String]
        let pleLayer = g["ple_layer"] as! Int
        let ng = g["ngram"] as! [String: Any]

        func hyper(_ prefix: String, withInjection: Bool) -> FNHyperConnection {
            let down = w.shape(prefix + "input_mix_weight_down.weight")
            return FNHyperConnection(
                hc: hc, hidden: hidden, lowrank: down[0], eps: eps,
                normWeight: w.data(prefix + "hc_norm.weight"), downWeight: w.data(prefix + "input_mix_weight_down.weight"),
                upWeight: w.data(prefix + "input_mix_weight_up.weight"),
                injectWeight: withInjection ? w.data(prefix + "block_inject_weight.weight") : [])
        }

        var layers: [FNDecoderLayer] = []
        for i in 0..<types.count {
            let base = "model.layers.\(i)."
            let mixer: FNMixer
            if types[i] == "linear_attention" {
                let p = base + "linear_attn."
                mixer = .linear(FNGatedDeltaNet(
                    hidden: hidden, kHeads: 2, vHeads: 4, kDim: 8, vDim: 8, convKernel: 4, eps: eps,
                    inProjQKV: w.data(p + "in_proj_qkv.weight"), inProjZ: w.data(p + "in_proj_z.weight"),
                    inProjB: w.data(p + "in_proj_b.weight"), inProjA: w.data(p + "in_proj_a.weight"),
                    convWeight: w.data(p + "conv1d.weight"), dtBias: w.data(p + "dt_bias"), aLog: w.data(p + "A_log"),
                    normWeight: w.data(p + "norm.weight"), outProj: w.data(p + "out_proj.weight")))
            } else {
                let p = base + "self_attn."
                let ix = p + "indexer."
                let indexer = FNQSAIndexer(heads: 2, kvHeads: 1, headDim: 8, budget: 8, compress: 4, eps: eps,
                                           qkProj: w.data(ix + "index_qk_proj.weight"),
                                           qNorm: w.data(ix + "q_layernorm.weight"), kNorm: w.data(ix + "k_layernorm.weight"))
                mixer = .full(FNAttention(
                    hidden: hidden, heads: 2, kvHeads: 1, headDim: 16, eps: eps,
                    qProj: w.data(p + "q_proj.weight"), kProj: w.data(p + "k_proj.weight"),
                    vProj: w.data(p + "v_proj.weight"), oProj: w.data(p + "o_proj.weight"),
                    qNorm: w.data(p + "q_norm.weight"), kNorm: w.data(p + "k_norm.weight"),
                    indexer: indexer, rope: FNRope(rotaryDim: 4, theta: 1e7)))
            }
            let m = base + "mlp."
            let moe = FNSparseMoE(
                hidden: hidden, experts: 8, topK: 2, intermediate: 12, sharedIntermediate: 12, normTopK: true,
                gate: w.data(m + "gate.weight"),
                source: FNArrayExperts(experts: 8, gateUp: w.data(m + "experts.gate_up_proj"),
                                       down: w.data(m + "experts.down_proj")),
                sharedGate: w.data(m + "shared_expert.gate_proj.weight"),
                sharedUp: w.data(m + "shared_expert.up_proj.weight"),
                sharedDown: w.data(m + "shared_expert.down_proj.weight"),
                sharedGateScalar: w.data(m + "shared_expert_gate.weight"))
            var ple: FNPLE? = nil
            var pleEmb: FNNgramEmbedding? = nil
            if i == pleLayer {
                let p = base + "ple."
                ple = FNPLE(hidden: hidden, hc: hc, embedDim: 8, convKernel: 4, dilation: 3, eps: eps,
                            keyProj: w.data(p + "key_proj.weight"), valueProj: w.data(p + "value_proj.weight"),
                            normKey: w.data(p + "norm_key.weight"), normQuery: w.data(p + "norm_query.weight"),
                            normConv: w.data(p + "norm_conv.weight"), convWeight: w.data(p + "conv1d.weight"))
                let table = FNGoldenTests.tensor(g["ple_table"])
                let headDim = (g["ple_table"] as! [String: Any])["shape"] as! [Int]
                pleEmb = FNNgramEmbedding(vocab: ng["vocab"] as! Int, ngram: ng["ngram_size"] as! Int,
                                          headsPerNgram: ng["heads_per_ngram"] as! Int,
                                          baseVocab: ng["base"] as! Int, seed: ng["seed"] as! Int,
                                          eos: Int64(ng["eos"] as! Int), pleLayerIndex: ng["ple_layer_index"] as! Int,
                                          rows: FNArrayRows(table: table, headDim: headDim[1]))
            }
            layers.append(FNDecoderLayer(mixer: mixer, attnHC: hyper(base + "attn_hyper_connection.", withInjection: true),
                                         mlpHC: hyper(base + "mlp_hyper_connection.", withInjection: true),
                                         moe: moe, ple: ple, pleEmbedding: pleEmb))
        }
        let model = FNModel(vocab: vocab, hidden: hidden, hc: hc,
                            embedding: w.data("model.embed_tokens.weight"), layers: layers,
                            finalMixer: hyper("model.hyper_connection_mixer.", withInjection: false),
                            lmHead: w.data("lm_head.weight"))
        return (model, g)
    }

    func testModelStreamsAndLogitsMatchHF() throws {
        let (model, g) = try Self.buildModel()
        let ids = (g["ids"] as! [[Int]]).map { $0.map { Int64($0) } }
        let streams = (g["streams"] as! [[String: Any]]).map { FNGoldenTests.tensor($0) }
        let logits = FNGoldenTests.tensor(g["logits"])
        let vocab = model.vocab
        let L = ids[0].count
        var gotStreams = [[Float]](repeating: [], count: streams.count)
        var gotLogits: [Float] = []
        for seq in ids {
            let (lg, st) = model.forwardTrace(seq)
            gotLogits += lg
            for (i, s) in st.enumerated() { gotStreams[i] += s }
        }
        for (i, want) in streams.enumerated() {
            let err = FNGoldenTests.maxAbsError(gotStreams[i], want)
            XCTAssertLessThan(err, 1e-3, "layer \(i) stream max err \(err)")
        }
        XCTAssertEqual(gotLogits.count, ids.count * L * vocab)
        XCTAssertLessThan(FNGoldenTests.maxAbsError(gotLogits, logits), 1e-3)
    }
}
