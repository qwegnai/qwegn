#!/usr/bin/env python3
"""Extract the pinned geometry golden from the HF config (C01 gate input).

Writes Scripts/oracle/geometry_golden.json — the single source of truth that
QWCore unit tests must match. Pinned to the fetched revision; never re-read
from `main` ad hoc.
"""
import json, hashlib, pathlib, sys

HERE = pathlib.Path(__file__).parent
cfg = json.loads((HERE / "config_golden.json").read_text())
t = cfg["text_config"]

golden = {
    "source": "Qwen/Qwen3.8-27B config.json",
    "sha256_config": hashlib.sha256((HERE / "config_golden.json").read_bytes()).hexdigest(),
    "model_type": cfg["model_type"],
    "hidden": t["hidden_size"],
    "vocab": t["vocab_size"],
    "intermediate": t["intermediate_size"],
    "num_layers": t["num_hidden_layers"],
    "full_attention_interval": t.get("full_attention_interval"),
    "q_heads": t["num_attention_heads"],
    "kv_heads": t["num_key_value_heads"],
    "head_dim": t["head_dim"],
    "q_proj_out": t["num_attention_heads"] * t["head_dim"] * (2 if t.get("attn_output_gate") else 1),
    "attn_output_gate": t.get("attn_output_gate"),
    "partial_rotary_factor": t.get("partial_rotary_factor"),
    "rope_theta": t.get("rope_parameters", {}).get("rope_theta"),
    "mrope_section": t.get("rope_parameters", {}).get("mrope_section"),
    "mrope_interleaved": t.get("rope_parameters", {}).get("mrope_interleaved"),
    "output_gate_type": t.get("output_gate_type"),
    "rms_eps": t.get("rms_norm_eps"),
    "max_position_embeddings": t.get("max_position_embeddings"),
    "gdn_qk_heads": t.get("linear_num_key_heads"),
    "gdn_v_heads": t.get("linear_num_value_heads"),
    "gdn_head_dim": t.get("linear_key_head_dim"),
    "gdn_conv_kernel": t.get("linear_conv_kernel_dim"),
    "mamba_ssm_dtype": t.get("mamba_ssm_dtype"),
    "layer_types_sample": t.get("layer_types", [])[:8],
    "num_mtp_layers": (t.get("mtp_num_hidden_layers") or t.get("num_mtp_layers") or 0),
    "mtp_use_dedicated_embeddings": t.get("mtp_use_dedicated_embeddings"),
}
# derived
golden["gdn_qkv_channels"] = (
    (golden["gdn_qk_heads"] * 2 + golden["gdn_v_heads"]) * golden["gdn_head_dim"]
)
golden["gdn_z_channels"] = golden["gdn_v_heads"] * golden["gdn_head_dim"]
golden["rotary_dims"] = int(golden["head_dim"] * golden["partial_rotary_factor"])

out = HERE / "geometry_golden.json"
out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
# show any keys we did not map, so nothing is silently dropped
mapped = set()
print(json.dumps({k: v for k, v in golden.items() if k != "layer_types_sample"}, indent=2))
