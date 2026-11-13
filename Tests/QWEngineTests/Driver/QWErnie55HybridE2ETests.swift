// QWErnie55HybridE2ETests.swift — Full-driver E2E parity tests for ERNIE-5.5 hybrid engine:
// HCA + Decoupled-RoPE MLA-DSA + Latent-MoE (SHVQ15_G64 + SHVQ3_G64 + HQ8_G32) + 4-stream mHC
// + MultiMax + shared-last-layer ECHO speculation + rollback + standalone `.qwcache`.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
@testable import QWEngine

final class QWErnie55HybridE2ETests: XCTestCase {

    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case hq8g32(n: Int, k: Int, seed: UInt32, scale: Float)
        case shvq(bits: Int, group: Int, n: Int, k: Int, seed: UInt32, scale: Float)
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func makeHQ8G32Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = scale * (1.0 + 0.125 * Float((r + g) % 5))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-127.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<group {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func makeSHVQData(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func detVec(_ count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    private static func buildSyntheticErnie55Pack(at rootDir: URL) throws -> QWModelDescriptor {
        let h = 256
        let v = 256
        let inter = 256
        let qHeads = 16
        let hcaHeadDim = 512
        let hcaNope = 448
        let hcaRope = 64
        let hcaQLora = 128
        let oGroups = 2
        let oLora = 128
        let compRatio = 4
        let window = 8

        let mlaQLora = 64
        let mlaKvLora = 512
        let mlaNope = 192
        let mlaRope = 64
        let mlaVDim = 256
        let idxHeads = 64
        let idxDim = 128

        let nExp = 32
        let topK = 4
        let moeInter = 64
        let moeLat = 64
        let streams = 4
        let mhcRows = 2 * streams + streams * streams

        let cfgJSON = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 2,
          "num_nextn_predict_layers": 1,
          "mtp_shared_last_layer": true,
          "max_position_embeddings": 16384,
          "rms_norm_eps": 1e-6,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": 1,
          "head_dim": \(hcaHeadDim),
          "qk_nope_head_dim": \(hcaNope),
          "qk_rope_head_dim": \(hcaRope),
          "v_head_dim": \(hcaHeadDim),
          "compress_ratios": [\(compRatio), -2, -2],
          "default_compress_ratio": \(compRatio),
          "q_lora_rank": \(hcaQLora),
          "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": \(hcaNope),
          "compress_qk_rope_head_dim": \(hcaRope),
          "compress_v_head_dim": \(hcaHeadDim),
          "o_groups": \(oGroups),
          "o_lora_rank": \(oLora),
          "sliding_window": \(window),
          "compress_rope_theta": 160000.0,
          "hca_rope_type": "yarn",
          "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 1024,
            "beta_fast": 32,
            "beta_slow": 1
          },
          "rope_theta": 8000000.0,
          "rotary_percent": 1.0,
          "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa",
          "hybrid_mla_num_attention_heads": \(qHeads),
          "hybrid_mla_num_key_value_heads": \(qHeads),
          "hybrid_mla_q_lora_rank": \(mlaQLora),
          "hybrid_mla_kv_lora_rank": \(mlaKvLora),
          "hybrid_mla_qk_nope_head_dim": \(mlaNope),
          "hybrid_mla_qk_rope_head_dim": \(mlaRope),
          "hybrid_mla_v_head_dim": \(mlaVDim),
          "mqa_split_kv_b_proj": true,
          "index_n_heads": \(idxHeads),
          "index_head_dim": \(idxDim),
          "index_topk": 64,
          "use_fast_hadamard": true,
          "use_vha_attention": true,
          "add_full_attention_sink_bias": true,
          "use_gated_attn": true,
          "use_attn_merger": true,
          "gated_attn_use_q_lora": true,
          "enable_hyper_connections": true,
          "hc_mult": \(streams),
          "hc_sinkhorn_iters": 20,
          "hc_eps": 1e-6,
          "hidden_act": "situ",
          "activation_situ_beta": 4.0,
          "activation_situ_linear_beta": 25.0,
          "first_k_dense_replace": 1,
          "moe_layer_freq": 1,
          "n_routed_experts": \(nExp),
          "num_experts_per_tok": \(topK),
          "moe_intermediate_size": \(moeInter),
          "moe_latent_size": \(moeLat),
          "latent_moe_use_norm": true,
          "moe_split_feature_routing": true,
          "routed_scaling_factor": 2.0,
          "routed_scaling_factor_learnable": true,
          "n_shared_experts": 1,
          "scoring_func": "sigmoid",
          "norm_topk_prob": true,
          "topk_method": "noaux_tc",
          "multimax": "lm_head",
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!

        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)

        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        let qlsaURL = assetsDir.appendingPathComponent("model.qlsa")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)

        let idxLsaDim = 64
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(
            inDim: hcaNope, indexDim: idxLsaDim, seed: 7)
        let qlsaProv = QWLSASidecar.Header.Provenance(
            modelId: "ernie5_5_moe",
            geometryDigest: descriptor.geometryDigestV2Hex,
            method: "weight_svd_v1",
            indexDim: idxLsaDim,
            poolSize: compRatio,
            recommendedBudget: 64,
            sinkTokens: compRatio,
            windowTokens: window
        )
        try QWLSASidecar.write(
            to: qlsaURL,
            provenance: qlsaProv,
            layerTensors: [0: (wq: basis, wk: basis)],
            wqShape: [idxLsaDim, hcaNope],
            wkShape: [idxLsaDim, hcaNope]
        )

        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL, qlsaURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = []
        planes.append(("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 101, amp: 0.25))))
        planes.append(("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 102, amp: 0.05, offset: 1.0))))
        planes.append(("lm_head.weight", .hq8g32(n: v, k: h, seed: 103, scale: 0.015)))
        planes.append(("model.mhc_contract.weight", .f32(shape: [streams, streams * h], values: detVec(streams * streams * h, seedInit: 104, amp: 0.01))))
        planes.append(("model.mhc_contract.bias", .f32(shape: [streams], values: [0.4, 0.3, 0.2, 0.1])))
        planes.append(("lm_head.multimax", .f32(shape: [8], values: [14.0, 15.0, 8.0, 0.5, -0.05, 0.25, -0.1, -0.02])))

        func appendMHC(_ blk: String, seedBase: UInt32) {
            planes.append((blk + "mhc_attn.weight", .f32(shape: [mhcRows, streams * h], values: detVec(mhcRows * streams * h, seedInit: seedBase, amp: 0.005))))
            planes.append((blk + "mhc_attn.bias", .f32(shape: [mhcRows], values: detVec(mhcRows, seedInit: seedBase + 1, amp: 0.05))))
            planes.append((blk + "mhc_ffn.weight", .f32(shape: [mhcRows, streams * h], values: detVec(mhcRows * streams * h, seedInit: seedBase + 2, amp: 0.005))))
            planes.append((blk + "mhc_ffn.bias", .f32(shape: [mhcRows], values: detVec(mhcRows, seedInit: seedBase + 3, amp: 0.05))))
        }

        func eyePlus(_ n: Int, seed: UInt32) -> [Float] {
            var m = detVec(n * n, seedInit: seed, amp: 0.01)
            for i in 0..<n { m[i * n + i] += 1.0 }
            return m
        }

        // Layer 0: HCA + Dense SiTU MLP + 4-stream mHC
        let b0 = "model.layers.0."
        planes.append((b0 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 200, amp: 0.05, offset: 1.0))))
        planes.append((b0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 201, amp: 0.05, offset: 1.0))))
        appendMHC(b0, seedBase: 210)
        planes.append((b0 + "self_attn.fused_in_proj.weight", .hq8g32(n: hcaQLora + 3 * hcaHeadDim, k: h, seed: 220, scale: 0.004)))
        planes.append((b0 + "self_attn.q_norm.weight", .bf16(shape: [hcaQLora], values: [Float](repeating: 1.0, count: hcaQLora))))
        planes.append((b0 + "self_attn.kv_norm.weight", .bf16(shape: [hcaHeadDim], values: [Float](repeating: 1.0, count: hcaHeadDim))))
        planes.append((b0 + "self_attn.comp_norm.weight", .bf16(shape: [hcaHeadDim], values: [Float](repeating: 1.0, count: hcaHeadDim))))
        planes.append((b0 + "self_attn.compressor.ape", .f32(shape: [compRatio, hcaHeadDim], values: detVec(compRatio * hcaHeadDim, seedInit: 225, amp: 0.02))))
        planes.append((b0 + "self_attn.q_up_gate_proj.weight", .hq8g32(n: qHeads * hcaHeadDim + oGroups * oLora, k: hcaQLora, seed: 226, scale: 0.004)))
        planes.append((b0 + "self_attn.wo_a.weight", .hq8g32(n: oGroups * oLora, k: (qHeads / oGroups) * hcaHeadDim, seed: 227, scale: 0.003)))
        planes.append((b0 + "self_attn.o_proj.weight", .hq8g32(n: h, k: oGroups * oLora, seed: 228, scale: 0.004)))
        planes.append((b0 + "self_attn.vha_mix.weight", .f32(shape: [qHeads, qHeads], values: eyePlus(qHeads, seed: 229))))
        planes.append((b0 + "self_attn.attn_sink", .f32(shape: [qHeads], values: detVec(qHeads, seedInit: 230, amp: 0.2, offset: -1.5))))
        planes.append((b0 + "mlp.gate_proj.weight", .hq8g32(n: inter, k: h, seed: 240, scale: 0.004)))
        planes.append((b0 + "mlp.up_proj.weight", .hq8g32(n: inter, k: h, seed: 241, scale: 0.004)))
        planes.append((b0 + "mlp.down_proj.weight", .hq8g32(n: h, k: inter, seed: 242, scale: 0.004)))

        // Layer 1: Decoupled-RoPE MLA-DSA + Latent-MoE (SHVQ15_G64 + SHVQ3_G64) + 4-stream mHC
        let b1 = "model.layers.1."
        planes.append((b1 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 300, amp: 0.05, offset: 1.0))))
        planes.append((b1 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 301, amp: 0.05, offset: 1.0))))
        appendMHC(b1, seedBase: 310)
        planes.append((b1 + "self_attn.q_a_proj.weight", .hq8g32(n: mlaQLora, k: h, seed: 320, scale: 0.004)))
        planes.append((b1 + "self_attn.kv_a_proj_with_mqa.weight", .hq8g32(n: mlaKvLora + mlaRope, k: h, seed: 321, scale: 0.004)))
        planes.append((b1 + "self_attn.indexer.wk.weight", .hq8g32(n: idxDim, k: h, seed: 322, scale: 0.004)))
        planes.append((b1 + "self_attn.q_a_layernorm.weight", .bf16(shape: [mlaQLora], values: [Float](repeating: 1.0, count: mlaQLora))))
        planes.append((b1 + "self_attn.kv_a_layernorm.weight", .bf16(shape: [mlaKvLora], values: [Float](repeating: 1.0, count: mlaKvLora))))
        planes.append((b1 + "self_attn.indexer.k_norm.weight", .bf16(shape: [idxDim], values: [Float](repeating: 1.0, count: idxDim))))
        planes.append((b1 + "self_attn.indexer.k_norm.bias", .bf16(shape: [idxDim], values: [Float](repeating: 0.0, count: idxDim))))
        planes.append((b1 + "self_attn.indexer.weights_proj.weight", .f32(shape: [idxHeads, h], values: detVec(idxHeads * h, seedInit: 325, amp: 0.02))))
        let qCh = qHeads * (mlaNope + mlaRope + mlaVDim)
        planes.append((b1 + "self_attn.q_b_gate_idx_proj.weight", .hq8g32(n: qCh + idxHeads * idxDim, k: mlaQLora, seed: 326, scale: 0.004)))
        planes.append((b1 + "self_attn.kv_b_proj.weight", .hq8g32(n: qHeads * (mlaNope + mlaVDim), k: mlaKvLora, seed: 327, scale: 0.004)))
        planes.append((b1 + "self_attn.o_proj.weight", .hq8g32(n: h, k: qHeads * mlaVDim, seed: 328, scale: 0.003)))
        planes.append((b1 + "self_attn.vha_mix.weight", .f32(shape: [qHeads, qHeads], values: eyePlus(qHeads, seed: 329))))
        planes.append((b1 + "self_attn.attn_sink", .f32(shape: [qHeads], values: detVec(qHeads, seedInit: 330, amp: 0.2, offset: -1.5))))

        planes.append((b1 + "mlp.gate.weight", .f32(shape: [2 * nExp, h], values: detVec(2 * nExp * h, seedInit: 340, amp: 0.08))))
        planes.append((b1 + "mlp.gate.e_score_correction_bias", .f32(shape: [nExp], values: detVec(nExp, seedInit: 341, amp: 0.1))))
        planes.append((b1 + "mlp.routed_scaling_factor_param", .f32(shape: [nExp], values: [Float](repeating: 1.5, count: nExp))))
        planes.append((b1 + "mlp.fc1.weight", .hq8g32(n: moeLat, k: h, seed: 342, scale: 0.004)))
        planes.append((b1 + "mlp.latent_norm.weight", .bf16(shape: [moeLat], values: [Float](repeating: 1.0, count: moeLat))))
        planes.append((b1 + "mlp.fc2.weight", .hq8g32(n: h, k: moeLat, seed: 343, scale: 0.004)))
        planes.append((b1 + "mlp.shared_experts.gate_proj.weight", .hq8g32(n: moeInter, k: h, seed: 344, scale: 0.004)))
        planes.append((b1 + "mlp.shared_experts.up_proj.weight", .hq8g32(n: moeInter, k: h, seed: 345, scale: 0.004)))
        planes.append((b1 + "mlp.shared_experts.down_proj.weight", .hq8g32(n: h, k: moeInter, seed: 346, scale: 0.004)))
        for e in 0..<nExp {
            let ep = b1 + "mlp.experts.\(e)."
            let sb = UInt32(400 + e * 10)
            planes.append((ep + "gate_proj.weight", .shvq(bits: 15, group: 64, n: moeInter, k: moeLat, seed: sb, scale: 0.03)))
            planes.append((ep + "up_proj.weight", .shvq(bits: 15, group: 64, n: moeInter, k: moeLat, seed: sb + 1, scale: 0.03)))
            planes.append((ep + "down_proj.weight", .shvq(bits: 3, group: 64, n: moeLat, k: moeInter, seed: sb + 2, scale: 0.03)))
        }

        // Layer 2: Shared-last-layer ECHO unique weights (aliases Layer 1 trunk & experts)
        let b2 = "model.layers.2."
        planes.append((b2 + "enorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append((b2 + "hnorm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append((b2 + "e_proj.weight", .hq8g32(n: h, k: h, seed: 500, scale: 0.004)))
        planes.append((b2 + "h_proj.weight", .hq8g32(n: h, k: h, seed: 501, scale: 0.004)))
        planes.append((b2 + "mhc_contract.weight", .f32(shape: [streams, streams * h], values: detVec(streams * streams * h, seedInit: 502, amp: 0.01))))
        planes.append((b2 + "mhc_contract.bias", .f32(shape: [streams], values: [0.25, 0.25, 0.25, 0.25])))
        planes.append((b2 + "shared_head.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append((b2 + "mlp.gate.e_score_correction_bias", .f32(shape: [nExp], values: detVec(nExp, seedInit: 503, amp: 0.12))))

        var file = Data()
        file.append(Data("QWP1".utf8))
        file.append(Data(repeating: 0, count: 8))
        var entries: [String: [String: Any]] = [:]

        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 {
                file.append(Data(repeating: 0, count: align - rem))
            }
        }
        padTo(256)

        for (name, spec) in planes {
            switch spec {
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { raw.append(contentsOf: $0) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                vals.withUnsafeBytes { (buf: UnsafeRawBufferPointer) in
                    file.append(contentsOf: buf)
                }
                padTo(256)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .hq8g32(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeHQ8G32Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = [
                    "shape": [n, k],
                    "hq4_off": off,
                    "hq": ["bits": 8, "group": 32],
                ]
            case .shvq(let bits, let group, let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeSHVQData(bits: bits, group: group, n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = [
                    "shape": [n, k],
                    "hq4_off": off,
                    "hq": ["bits": bits, "group": group, "vq": "shvq1"],
                ]
            }
        }

        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "ernie5_5_moe",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-ernie55-e2e",
                "sha256_source": "synthetic",
                "quant_recorded": "dense_hq8_g32__expert_shvq15_g64__expdown_shvq3_g64",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { raw in
            file.replaceSubrange(4..<12, with: raw)
        }
        file.append(hdrData)
        let qwpURL = rootDir.appendingPathComponent("model.qwp")
        try file.write(to: qwpURL)
        return descriptor
    }

    private func runGreedyTokens(driver: QWEngineDriver, prompt: [Int], maxNew: Int, speculative: Bool) throws -> [Int] {
        let session = try driver.makeSession(promptTokens: prompt.count)
        defer { driver.finishSession(session) }
        if speculative {
            return try driver.generateExperimentalGreedy(session: session, prompt: prompt, maxNew: maxNew)
        }
        return try driver.generate(session: session, prompt: prompt, maxNew: maxNew)
    }

    func testGreedyVsEchoSpeculativeBitExactTokenStreamParity() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-ernie55-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        _ = try Self.buildSyntheticErnie55Pack(at: rootDir)

        let prompt = [1, 7, 19, 42, 7, 19, 42, 88, 103, 7, 19, 42, 155, 201]
        let maxNew = 10

        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertEqual(baseDriver.weightFormat, .hqBlock(
            bits: 8, groupSize: 32,
            expertBits: 15, expertGroupSize: 64,
            denseIncumbent: false, expertIncumbent: false,
            expertDownBits: 3, expertDownGroupSize: 64,
            expertDownIncumbent: false,
            denseSHVQ: false, expertSHVQ: true, expertDownSHVQ: true))

        let baselineTokens = try runGreedyTokens(
            driver: baseDriver, prompt: prompt, maxNew: maxNew, speculative: false)
        XCTAssertEqual(baselineTokens.count, maxNew)

        let echo2Driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 2, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "mtp", prefillChunkSize: 32)
        let echo2Tokens = try runGreedyTokens(
            driver: echo2Driver, prompt: prompt, maxNew: maxNew, speculative: true)
        XCTAssertEqual(echo2Tokens, baselineTokens,
                       "2-draft shared-last-layer ECHO speculative stream must be bit-exact with greedy baseline")

        let hybrid3Driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "hybrid", prefillChunkSize: 32)
        let hybrid3Tokens = try runGreedyTokens(
            driver: hybrid3Driver, prompt: prompt, maxNew: maxNew, speculative: true)
        XCTAssertEqual(hybrid3Tokens, baselineTokens,
                       "3-draft hybrid PLD + ECHO speculative stream must be bit-exact with greedy baseline")
    }

    func testPrivatePrefixRollbackAndStandaloneQWCacheBitExactParity() throws {
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-ernie55-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        _ = try Self.buildSyntheticErnie55Pack(at: rootDir)

        let prefixTokens = [1, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143]
        let suffixTokens = [154, 165]
        let fullPrompt = prefixTokens + suffixTokens
        let maxNew = 6

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let expected = try runGreedyTokens(driver: driver, prompt: fullPrompt, maxNew: maxNew, speculative: false)

        // 1a. In-place session snapshot & rollback mid-session
        let s = try driver.makeSession(promptTokens: fullPrompt.count)
        let firstTok = try driver.prefill(s, tokens: fullPrompt)
        let inPlaceSnap = try driver.snapshotSession(s)
        let privatePrefix = try driver.capturePrivatePrefix(s, maxBytes: 64 << 20)
        _ = try driver.decodeStep(s, token: firstTok)
        _ = try driver.decodeStep(s, token: 5)
        try driver.rollbackSession(s, to: inPlaceSnap)
        var rolledBack = [firstTok]
        var cur = firstTok
        for _ in 1..<maxNew {
            cur = try driver.decodeStep(s, token: cur)
            rolledBack.append(cur)
        }
        driver.finishSession(s)
        XCTAssertEqual(rolledBack, expected,
                       "In-place session rollback over HCA + MLA-DSA q4 state must match fresh greedy decode")

        // 1b. Private prefix adoption into a fresh session
        let s2 = try driver.makeSession(promptTokens: fullPrompt.count)
        defer { driver.finishSession(s2) }
        try driver.adoptPrivatePrefix(privatePrefix, into: s2)
        var adoptedOut = [firstTok]
        var cur2 = firstTok
        for _ in 1..<maxNew {
            cur2 = try driver.decodeStep(s2, token: cur2)
            adoptedOut.append(cur2)
        }
        XCTAssertEqual(adoptedOut, expected,
                       "Private prefix adoption over HCA + MLA-DSA q4 state must match fresh greedy decode")

        // 2a. Standalone `.qwcache` non-boundary open HCA block (`14 + 2`, 2 open compressor slots) vs explicit 2-chunk prefill
        let sTwoChunk = try driver.makeSession(promptTokens: fullPrompt.count)
        _ = try driver.prefill(sTwoChunk, tokens: prefixTokens)
        sTwoChunk.generated = 0
        let firstTwoChunk = try driver.prefill(sTwoChunk, tokens: suffixTokens)
        var twoChunkExpected = [firstTwoChunk]
        var curTwo = firstTwoChunk
        for _ in 1..<maxNew {
            curTwo = try driver.decodeStep(sTwoChunk, token: curTwo)
            twoChunkExpected.append(curTwo)
        }
        driver.finishSession(sTwoChunk)

        let qwcacheURL = rootDir.appendingPathComponent("prefix.qwcache")
        let writtenBytes = try driver.buildStandalonePrefixCache(
            tokens: prefixTokens, outputURL: qwcacheURL, pinAsSharedBase: false)
        XCTAssertGreaterThan(writtenBytes, 0)

        let freshDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let mounted = try freshDriver.mountStandalonePrefixCache(from: qwcacheURL)
        XCTAssertEqual(mounted.tokens, prefixTokens)
        let cachedOut = try freshDriver.generateCachedGreedy(prompt: fullPrompt, maxNew: maxNew)
        XCTAssertEqual(cachedOut, twoChunkExpected,
                       "Mounted standalone .qwcache suffix continuation over open HCA + MLA-DSA q4 state must be bit-exact with 2-chunk prefill")

        // 2b. Standalone `.qwcache` full shipping chunk (`32 + 6` tokens) vs `driver.generate` (`prefillChunkSize = 32`)
        let prefix32 = (0..<32).map { ($0 * 7 + 3) % 251 + 1 }
        let suffix6 = [19, 41, 83, 127, 163, 211]
        let prompt38 = prefix32 + suffix6
        let expected38 = try runGreedyTokens(driver: driver, prompt: prompt38, maxNew: maxNew, speculative: false)
        let cache32URL = rootDir.appendingPathComponent("prefix32.qwcache")
        XCTAssertGreaterThan(try driver.buildStandalonePrefixCache(
            tokens: prefix32, outputURL: cache32URL, pinAsSharedBase: false), 0)
        let fresh32Driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        _ = try fresh32Driver.mountStandalonePrefixCache(from: cache32URL)
        let cached38Out = try fresh32Driver.generateCachedGreedy(prompt: prompt38, maxNew: maxNew)
        XCTAssertEqual(cached38Out, expected38,
                       "Mounted 32-token .qwcache + 6-token suffix must be bit-exact with standard 32-chunked generate")

        // 2c. Standalone `.qwcache` exact-hit continuation (`16` tokens) vs single-chunk prefill
        let exactCacheURL = rootDir.appendingPathComponent("exact16.qwcache")
        let exactBytes = try driver.buildStandalonePrefixCache(
            tokens: fullPrompt, outputURL: exactCacheURL, pinAsSharedBase: false)
        XCTAssertGreaterThan(exactBytes, 0)
        let exactFreshDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        _ = try exactFreshDriver.mountStandalonePrefixCache(from: exactCacheURL)
        let exactCachedOut = try exactFreshDriver.generateCachedGreedy(prompt: fullPrompt, maxNew: maxNew)
        XCTAssertEqual(exactCachedOut, expected,
                       "Mounted standalone .qwcache exact hit must be bit-exact with single-chunk prefill")
    }
}
