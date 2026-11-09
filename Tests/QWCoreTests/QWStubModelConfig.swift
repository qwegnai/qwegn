import Foundation
@testable import QWCore

/// Stub model configs for the unit suite: no unit test may read an installed
/// model pack or its assets, so the suite runs on a machine with no models.
enum QWStubModelConfig {
    /// The dense 27B's `text_config`, as the pack states it.
    static func dense27BText() -> [String: Any] {
        var text: [String: Any] = baseText(
            modelType: "qwen3_5_text",
            hidden: 5120,
            vocab: 248320,
            numLayers: 64,
            fullAttentionInterval: 4,
            qHeads: 24,
            kvHeads: 4,
            headDim: 256,
            intermediateSize: 17408,
            attnOutputGate: true,
            outputGateType: "swish"
        )
        text["linear_key_head_dim"] = 128
        text["linear_num_key_heads"] = 16
        text["linear_num_value_heads"] = 48
        text["linear_value_head_dim"] = 128
        text["linear_conv_kernel_dim"] = 4
        text["mtp_num_hidden_layers"] = 1
        text["mtp_use_dedicated_embeddings"] = false
        return text
    }

    /// The Flash-Next `text_config`, as the pack states it.
    static func flashNextText() -> [String: Any] {
        var text: [String: Any] = baseText(
            modelType: "qwen4_exp_text",
            hidden: 2560,
            vocab: 248320,
            numLayers: 48,
            fullAttentionInterval: 4,
            qHeads: 24,
            kvHeads: 2,
            headDim: 256,
            intermediateSize: nil,
            attnOutputGate: nil,
            outputGateType: "sigmoid"
        )
        text["linear_key_head_dim"] = 128
        text["linear_num_key_heads"] = 16
        text["linear_num_value_heads"] = 48
        text["linear_value_head_dim"] = 128
        text["linear_conv_kernel_dim"] = 4
        text["num_experts"] = 512
        text["num_experts_per_tok"] = 10
        text["moe_intermediate_size"] = 640
        text["shared_expert_intermediate_size"] = 640
        text["indexer_n_heads"] = 4
        text["indexer_kv_heads"] = 1
        text["indexer_head_dim"] = 128
        text["indexer_budget"] = 2048
        text["indexer_compress_ratio"] = 4
        text["hc_count"] = 4
        text["hc_lowrank"] = 320
        text["ple_layer_ids"] = [2]
        text["ple_embed_dim"] = 2560
        text["ple_conv_kernel_size"] = 4
        text["ngram_size"] = 3
        text["heads_per_ngram"] = 8
        text["ngram_vocab_size_base"] = 20_000_000
        text["make_ngram_vocab_size_divisible_by"] = 128
        text["split_ngram_parts"] = 128
        text["mtp_num_hidden_layers"] = 1
        text["mtp_use_dedicated_embeddings"] = false
        return text
    }

    /// The dense 27B's full config body.
    static func dense27BJSON() throws -> Data {
        try envelope(text: dense27BText(), modelType: "qwen3_5")
    }

    /// The Flash-Next full config body.
    static func flashNextJSON() throws -> Data {
        try envelope(text: flashNextText(), modelType: "qwen4_exp")
    }

    /// The dense 27B's geometry, decoded from the stub config.
    static func dense27BGeometry() throws -> QWModelGeometry {
        QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: try dense27BJSON()))
    }

    /// The Flash-Next geometry, decoded from the stub config.
    static func flashNextGeometry() throws -> QWModelGeometry {
        QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: try flashNextJSON()))
    }

    private static func baseText(modelType: String, hidden: Int, vocab: Int, numLayers: Int,
                                 fullAttentionInterval: Int, qHeads: Int, kvHeads: Int,
                                 headDim: Int, intermediateSize: Int?, attnOutputGate: Bool?,
                                 outputGateType: String?) -> [String: Any] {
        var text: [String: Any] = [
            "model_type": modelType,
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": numLayers,
            "layer_types": layerTypes(count: numLayers, interval: fullAttentionInterval),
            "full_attention_interval": fullAttentionInterval,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "partial_rotary_factor": 0.25,
            "rope_parameters": [
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
            ],
            "eos_token_id": 248044,
            "bos_token_id": 248044,
        ]
        if let intermediateSize {
            text["intermediate_size"] = intermediateSize
        }
        if let attnOutputGate {
            text["attn_output_gate"] = attnOutputGate
        }
        if let outputGateType {
            text["output_gate_type"] = outputGateType
        }
        return text
    }
    private static func layerTypes(count: Int, interval: Int) -> [String] {
        (0..<count).map { ($0 + 1) % interval == 0 ? "full_attention" : "linear_attention" }
    }

    private static func envelope(text: [String: Any], modelType: String) throws -> Data {
        let body: [String: Any] = [
            "model_type": modelType,
            "text_config": text,
            "vision_config": [
                "depth": 27,
                "hidden_size": 1152,
                "num_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
            ],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        ]
        return try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    }
}
