// QWModelDescriptor+ConfigEnvelope.swift — Decodable HF/Qwen config JSON schema.
import Foundation

extension QWModelDescriptor {
    internal final class ConfigEnvelope: Decodable {
        final class RopeParameters: Decodable {
            let mrope_section: [Int]?
            let mrope_interleaved: Bool?
            let rope_theta: Double?
            let partial_rotary_factor: Double?
            let type: String?
            let rope_type: String?
            let factor: Double?
            let original_max_position_embeddings: Int?
            let beta_fast: Double?
            let beta_slow: Double?
            let mscale: Double?
            let mscale_all_dim: Double?
            var full_attention: RopeParameters? = nil
            var sliding_attention: RopeParameters? = nil
        }
        final class RopeScaling: Decodable {
            let type: String?
            let rope_type: String?
            let factor: Double?
            let original_max_position_embeddings: Int?
            let beta_fast: Double?
            let beta_slow: Double?
            let mscale: Double?
            let mscale_all_dim: Double?
        }
        final class VisionConfig: Decodable {
            let depth: Int?
            var num_hidden_layers: Int? = nil
            let hidden_size: Int?
            let num_heads: Int?
            var num_attention_heads: Int? = nil
            let intermediate_size: Int?
            let patch_size: Int?
            let temporal_patch_size: Int?
            let spatial_merge_size: Int?
            let num_position_embeddings: Int?
            var num_patches: Int? = nil
            let out_hidden_size: Int?
            let image_size: Int?
            let rms_norm_eps: Double?
            var norm_eps: Double? = nil
            var layer_norm_eps: Double? = nil
            let hidden_act: String?
            let model_type: String?
            var disable_merger_proj: Bool? = nil
            var num_key_value_heads: Int? = nil
            var layer_types: [String]? = nil
            var conv_kernel_size: Int? = nil
            var downsample_factor: Int? = nil
            var out_dim: Int? = nil
            var rope_theta: Double? = nil
            var rope_parameters: RopeParameters? = nil
        }
        struct GenerationConfig: Decodable {
            let eos_token_id: JSONIntOrList?
            let pad_token_id: Int?
        }
        struct RawLoopSettings: Decodable {
            let start: Int?
            let end: Int?
            let eval_count: Int?
            let counts: [Int]?
        }
        struct RawMHCConfig: Decodable {
            let start: Int?
            let end: Int?
            let streams: Int?
            let iterations: Int?
            let fixed_identity: Bool?
            let pre_scale: Float?
        }
        enum MHCValue: Decodable {
            case flag(Bool)
            case config(RawMHCConfig)

            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let b = try? c.decode(Bool.self) {
                    self = .flag(b)
                    return
                }
                self = .config(try c.decode(RawMHCConfig.self))
            }
        }
        struct RawLinearAttnConfig: Decodable {
            let Linear_type: String?
            let linear_type: String?
            let num_heads: Int?
            let head_dim: Int?
            let short_conv_kernel_size: Int?
            let gate_lower_bound: Float?
            let low_rank_dim: Int?
        }
        final class Text: Decodable {
            var model_type: String? = nil
            let hidden_size: Int
            let vocab_size: Int
            let num_hidden_layers: Int
            var layer_types: [String]? = nil
            var layer_group_size: Int? = nil
            var compress_ratios: [Int]? = nil
            var full_attention_interval: Int? = nil
            var full_attention_layers: [Int]? = nil
            var dense_attention_layers: [Int]? = nil
            var rms_norm_eps: Double? = nil
            var norm_eps: Double? = nil
            var layer_norm_eps: Double? = nil
            var block_norm_eps: Double? = nil
            var norm_bias: Bool? = nil
            var norm_type: String? = nil
            var indexer_norm_type: String? = nil
            var layernorm_type: String? = nil
            var layernorm_gating_weight: Float? = nil
            var use_sandwich_norm: Bool? = nil
            var conv_L_cache: Int? = nil
            var conv_bias: Bool? = nil
            var block_auto_adjust_ff_dim: Bool? = nil
            var block_ffn_dim_multiplier: Double? = nil
            var block_multiple_of: Int? = nil
            var global_rope_theta: Double? = nil
            var local_rope_theta: Double? = nil
            var local_attention: Int? = nil
            var global_attn_every_n_layers: Int? = nil
            var classifier_pooling: String? = nil
            var id2label: [String: String]? = nil
            var num_labels: Int? = nil
            var pad_token_id: Int? = nil
            var bos_token_id: Int? = nil
            var max_position_embeddings: Int? = nil
            var model_max_length: Int? = nil
            let num_attention_heads: Int
            var num_key_value_heads: Int? = nil
            var architectures: [String]? = nil
            var head_dim: Int? = nil
            var attention_head_size: Int? = nil
            var qk_head_dim: Int? = nil
            var attention_variant: String? = nil
            var block_sparse_gqa_indexer_n_heads: Int? = nil
            var block_sparse_gqa_indexer_head_dim: Int? = nil
            var block_sparse_gqa_block_size: Int? = nil
            var block_sparse_gqa_prefix_window_size: Int? = nil
            var block_sparse_gqa_local_window_size: Int? = nil
            var block_sparse_gqa_num_remote_blocks: Int? = nil
            var qk_layernorm: Bool? = nil
            var use_qk_norm: Bool? = nil
            var qk_norm: Bool? = nil
            var attention_bias: Bool? = nil
            var qkv_bias: Bool? = nil
            var mlp_bias: Bool? = nil
            var output_router_logits: Bool? = nil
            var qk_nope_head_dim: Int? = nil
            var qk_rope_head_dim: Int? = nil
            var v_head_dim: Int? = nil
            var kv_lora_rank: Int? = nil
            var q_lora_rank: Int? = nil
            var o_groups: Int? = nil
            var o_lora_rank: Int? = nil
            var sliding_window: Int? = nil
            var use_sliding_window: Bool? = nil
            var max_window_layers: Int? = nil
            var sliding_window_pattern: Int? = nil
            var original_max_position_embeddings: Int? = nil
            var mla_use_nope: Bool? = nil
            var use_mla_nope: Bool? = nil
            var use_mla: Bool? = nil
            var partial_rotary_factor: Double? = nil
            var rotary_percent: Double? = nil
            var rotary_dim: Int? = nil
            let rope_theta: Double?
            var compress_rope_theta: Double? = nil
            var rope_parameters: RopeParameters? = nil
            var rope_scaling: RopeScaling? = nil
            var rope_type: String? = nil
            var hca_rope_type: String? = nil
            var rope_interleave: Bool? = nil
            let output_gate_type: String?
            var attn_output_gate: Bool? = nil
            var use_gated_attn: Bool? = nil
            var gated_attention: Bool? = nil
            var gated_attention_proj_granularity_type: String? = nil
            var use_mla_scaling_factor: Bool? = nil
            var use_attn_merger: Bool? = nil
            var use_vha_attention: Bool? = nil
            var add_full_attention_sink_bias: Bool? = nil
            var gated_attn_use_q_lora: Bool? = nil
            var hybrid_mla_attention: String? = nil
            var hybrid_mla_num_attention_heads: Int? = nil
            var hybrid_mla_q_lora_rank: Int? = nil
            var hybrid_mla_kv_lora_rank: Int? = nil
            var hybrid_mla_qk_nope_head_dim: Int? = nil
            var hybrid_mla_qk_rope_head_dim: Int? = nil
            var hybrid_mla_v_head_dim: Int? = nil
            var use_fast_hadamard: Bool? = nil
            var mqa_split_kv_b_proj: Bool? = nil
            let linear_key_head_dim: Int?
            let linear_num_key_heads: Int?
            let linear_num_value_heads: Int?
            let linear_value_head_dim: Int?
            let linear_conv_kernel_dim: Int?
            var linear_attention_type: String? = nil
            var linear_gating_type: String? = nil
            var linear_sigmoid_gate_scale: Float? = nil
            var linear_attn_o_norm_eps: Double? = nil
            var num_kv_heads_for_linear_attn: Int? = nil
            var short_conv_kernel_size: Int? = nil
            var no_kda_lora: Bool? = nil
            var use_kda_lora: Bool? = nil
            var kda_safe_gate: Bool? = nil
            var kda_lower_bound: Float? = nil
            var linear_silu: Bool? = nil
            var linear_attn_config: RawLinearAttnConfig? = nil
            let num_experts: Int?
            var n_routed_experts: Int? = nil
            let num_experts_per_tok: Int?
            let moe_intermediate_size: Int?
            var moe_latent_size: Int? = nil
            var latent_moe_use_norm: Bool? = nil
            let shared_expert_intermediate_size: Int?
            var moe_shared_expert_intermediate_size: Int? = nil
            var n_shared_experts: Int? = nil
            var num_shared_experts: Int? = nil
            var moe_shared_expert_gate: Bool? = nil
            var use_shared_expert_gate: Bool? = nil
            var use_shared_expert_sigmoid: Bool? = nil
            var moe_split_feature_routing: Bool? = nil
            var moe_router_enable_expert_bias: Bool? = nil
            var n_group: Int? = nil
            var topk_group: Int? = nil
            var first_k_dense_replace: Int? = nil
            var mlp_only_layers: [Int]? = nil
            var decoder_sparse_step: Int? = nil
            var moe_layer_freq: Int? = nil
            var routed_scaling_factor: Float? = nil
            var routed_scaling_factor_learnable: Bool? = nil
            var scoring_func: String? = nil
            var score_function: String? = nil
            var router_score_function: String? = nil
            var topk_method: String? = nil
            var mlp_clamp: Bool? = nil
            var swiglu_limit: Float? = nil
            var swiglu_limits: Float? = nil
            var expert_swiglu_limit_list: [Float]? = nil
            var share_expert_swiglu_limit_list: [Float]? = nil
            var norm_topk_prob: Bool? = nil
            var hidden_act: String? = nil
            var activation_situ_beta: Float? = nil
            var activation_situ_linear_beta: Float? = nil
            let indexer_budget: Int?
            var index_topk: Int? = nil
            let indexer_compress_ratio: Int?
            var index_kpool: Int? = nil
            let indexer_head_dim: Int?
            var index_head_dim: Int? = nil
            let indexer_kv_heads: Int?
            let indexer_n_heads: Int?
            var index_n_heads: Int? = nil
            var index_kpool_compress: Bool? = nil
            var index_kpool_always_select_tail: Bool? = nil
            var index_share_for_mtp_iteration: Bool? = nil
            var indexer_rope_interleave: Bool? = nil
            let hc_count: Int?
            let hc_lowrank: Int?
            var enable_hyper_connections: Bool? = nil
            var num_residual_streams: Int? = nil
            var mhc_sinkhorn_iterations: Int? = nil
            var hc_mult: Int? = nil
            var hc_sinkhorn_iters: Int? = nil
            var hc_eps: Float? = nil
            let ple_layer_ids: [Int]?
            let ple_embed_dim: Int?
            let ple_conv_kernel_size: Int?
            let ngram_size: Int?
            let heads_per_ngram: Int?
            let ngram_vocab_size_base: Int?
            let make_ngram_vocab_size_divisible_by: Int?
            let split_ngram_parts: Int?
            let mtp_num_hidden_layers: Int?
            var num_nextn_predict_layers: Int? = nil
            var mtp_num_layers: Int? = nil
            var nextn_is_sparse: Bool? = nil
            var nextn_is_linear: Bool? = nil
            let mtp_use_dedicated_embeddings: Bool?
            var mtp_shared_last_layer: Bool? = nil
            var final_logit_softcapping: Float? = nil
            var dspark_block_size: Int? = nil
            var dspark_markov_rank: Int? = nil
            var dspark_noise_token_id: Int? = nil
            var dspark_target_layer_ids: [Int]? = nil
            var dspark_num_hidden_layers: Int? = nil
            var dspark_sliding_window: Int? = nil
            var dspark_rope_theta: Double? = nil
            var enable_confidence_head: Bool? = nil
            var multimax: String? = nil
            var tie_word_embeddings: Bool? = nil
            var eos_token_id: JSONIntOrList? = nil
            let intermediate_size: Int?           // dense models / first_k_dense_replace
            var quantization: QWQuantizationSpec? = nil
            var quantization_config: QWQuantizationSpec? = nil
            var loop_settings: RawLoopSettings? = nil
            var mhc: MHCValue? = nil
        }
        let text_config: Text?
        let vision_config: VisionConfig?
        let generation_config: GenerationConfig?
        let eos_token_id: JSONIntOrList?
        let max_position_embeddings: Int?
        let model_max_length: Int?
        let quantization: QWQuantizationSpec?
        let quantization_config: QWQuantizationSpec?
        let image_token_id: Int?
        let video_token_id: Int?
        let vision_start_token_id: Int?
        let vision_end_token_id: Int?
        let model_type: String?
        let architectures: [String]?
        let tie_word_embeddings: Bool?
        let original_max_position_embeddings: Int?
        let mrope_section: [Int]?
        let loop_settings: RawLoopSettings?
        let mhc: MHCValue?
        let max_length: Int?
        let image_text_length: Int?
        let audio_text_length: Int?
        let head_layers: Int?
        let temperatures: [String: Double]?
        let pad_token_id: Int?
        let bos_token_id: Int?
        let d1_codes: Int?
        let d1_attention: String?
        let d1_numbers: Bool?
        let d1_set_size: Int?
        let d1_vision_max_pixels: Int?
        let projector_hidden_size: Int?
        let projector_layer_norm_eps: Double?
        let projector_bias: Bool?
        let auto_map: [String: String]?
        let id2label: [String: String]?
        let classifier_pooling: String?
        let num_labels: Int?
    }

}
