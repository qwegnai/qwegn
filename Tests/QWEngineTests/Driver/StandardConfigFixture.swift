// StandardConfigFixture.swift — a standard model's geometry, built through
// the same decoder a real checkpoint takes.
import Foundation
import XCTest
@testable import QWCore

extension QWModelGeometry {
    /// A pure-attention standard geometry, built through the **same decoder
    /// path a real checkpoint takes** — so the fixture cannot drift from the
    /// schema the standard decoder implements.
    static func stubPureAttention(gated: Bool = false) throws -> QWModelGeometry {
        var config: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 2048,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 151645,
            "vocab_size": 151936,
        ]
        if gated { config["output_gate_type"] = "sigmoid" }
        let data = try JSONSerialization.data(withJSONObject: config)
        let descriptor = try QWModelDescriptor.decode(configJSON: data)
        return QWModelGeometry(descriptor: descriptor)
    }

    /// A standard dense geometry whose head the attention kernels serve
    /// today: a 256-half head, no gate, no q/k norm.
    /// A standard dense geometry at an arbitrary shape, for the law's grid.
    static func stubShape(kvHeads: Int, qHeads: Int, headDim: Int) throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A sparse MoE geometry: 32 experts (the router's law is
    /// `experts % 32 == 0`), top-8, and a 256-wide expert intermediate.
    /// `normTopKProb` is the model's own top-k normalisation flag.
    static func stubPureMoE(normTopKProb: Bool = false) throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "num_experts": 32,
            "num_experts_per_tok": 8,
            "norm_topk_prob": normTopKProb,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    static func stubStandard256Geometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 32000,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A standard dense geometry in the llama shape: a 128-half head, no
    /// attention output gate, no q/k norm.
    static func stubLlamaGeometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 32000,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A one-layer qwen3 small enough to pack in a test: the same schema
    /// and the same decoder as a published checkpoint. Its head is 128
    /// halves, and it has **two** kv heads, so the k norm's tiling runs.
    static func stubTinyQwen3Geometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    /// A one-layer llama small enough to pack in a test: the same schema
    /// and the same decoder, a fraction of the size.
    static func stubTinyLlamaGeometry() throws -> QWModelGeometry {
        let config: [String: Any] = [
            "model_type": "llama",
            "hidden_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }
}
