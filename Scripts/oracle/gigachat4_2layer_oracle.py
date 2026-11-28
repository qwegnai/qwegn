# gigachat4_2layer_oracle.py — 2-layer + NextN MTP PyTorch-to-Metal parity oracle for GigaChat4.
# Packs a synthetic hybrid (L0 Dense+GDN, L1 MoE+DSA-MLA, L2 NextN Dense+MLA) checkpoint via
# gigachat4_adapter.py (hq8_g32), reloads exact QWP1 weights into modeling_gigachat4.py, and emits goldens.
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "quantize"))

from gigachat4_adapter import quantize_gigachat4_model  # noqa: E402
from hq_blocks import dequant_hq8_g32_m5  # noqa: E402
from shvq import any_block_bytes, dequant_any_blocks  # noqa: E402
from verify import read_header  # noqa: E402

DEFAULT_HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--ai-sage--GigaChat4-37B-A4B"
    / "snapshots/1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc"
)


def _bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _load_hf_gigachat4_modules(snapshot_dir: Path):
    pkg_name = "_qwg_hf_gigachat4"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(snapshot_dir)]
        sys.modules[pkg_name] = pkg
    cfg_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.configuration_gigachat4", snapshot_dir / "configuration_gigachat4.py"
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules[cfg_spec.name] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    mod_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_gigachat4", snapshot_dir / "modeling_gigachat4.py"
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_spec.name] = mod
    mod_spec.loader.exec_module(mod)
    return cfg_mod, mod


def _patch_bf16_gated_norm(mod) -> None:
    def _gated_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out_dt = hidden_states.dtype
        xf = hidden_states.float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        h = _bf16_round((xf * rstd) * self.weight.float())
        g1 = _bf16_round(F.silu(F.linear(h, self.gate_up_projection.weight.float())))
        g2 = _bf16_round(torch.sigmoid(F.linear(g1, self.gate_down_projection.weight.float())))
        y = _bf16_round((h * float(self.layernorm_gating_weight)) * g2)
        return y.to(out_dt)

    mod.GigaChat4GatedNorm.forward = _gated_forward


def _build_synthetic_config_dict() -> dict:
    return {
        "architectures": ["GigaChat4ForCausalLM"],
        "model_type": "gigachat4",
        "vocab_size": 1024,
        "max_position_embeddings": 32768,
        "hidden_size": 256,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "nextn_is_sparse": False,
        "nextn_is_linear": False,
        "first_k_dense_replace": 1,
        "full_attention_layers": [1],
        "dense_attention_layers": [],
        "linear_attention_type": "GigaChat4GatedDeltaNet",
        "linear_gating_type": "gated_rmsnorm_sigmoid",
        "linear_sigmoid_gate_scale": 2.0,
        "linear_attn_o_norm_eps": 1e-6,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_use_short_conv": True,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_krope_heads": 1,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_head_dim": 192,
        "head_dim": 64,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 4,
        "indexer_norm_type": "LlamaRMSNorm",
        "indexer_rotate_activation_enabled": None,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "use_shared_expert_sigmoid": False,
        "norm_type": "GatedNorm",
        "layernorm_type": "pre_post",
        "layernorm_gating_weight": 2.0,
        "rms_norm_eps": 1e-6,
        "gated_attention": True,
        "use_mla_scaling_factor": True,
        "swiglu_limit": 10.0,
        "final_logit_softcapping": 30.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_theta": 100000.0,
        "rope_interleave": True,
        "rope_scaling": {
            "type": "yarn",
            "factor": 32.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32768,
        },
        "tie_word_embeddings": False,
        "use_cache": False,
        "bos_token_id": 1,
        "pad_token_id": 2,
        "eos_token_id": 2,
    }


class _NextNMTPModule(nn.Module):
    def __init__(self, cfg_mod, mod, tc: dict):
        super().__init__()
        mtp_tc = dict(tc)
        mtp_tc["num_hidden_layers"] = 1
        mtp_tc["first_k_dense_replace"] = 1
        mtp_tc["full_attention_layers"] = [0]
        mtp_tc["dense_attention_layers"] = [0]
        mtp_cfg = cfg_mod.GigaChat4Config(**mtp_tc)
        mtp_cfg._attn_implementation = "eager"
        self.mtp_cfg = mtp_cfg
        h = int(tc["hidden_size"])
        self.enorm = mod.build_norm(mtp_cfg, h)
        self.hnorm = mod.build_norm(mtp_cfg, h)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        self.layer = mod.GigaChat4DecoderLayer(mtp_cfg, layer_idx=0)
        self.shared_head_norm = mod.build_norm(mtp_cfg, h)
        self.rotary_emb = mod.GigaChat4RotaryEmbedding(config=mtp_cfg)
        self.cache_cls = mod.GigaChat4DynamicCache
        self.create_causal_mask = mod.create_causal_mask
        self.uses_inputs_embeds = mod._CAUSAL_MASK_USES_INPUTS_EMBEDS


def _init_deterministic_weights(model: nn.Module, nextn: _NextNMTPModule, mod, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for m in (model, nextn):
            for name, param in m.named_parameters():
                if name.endswith("A_log"):
                    u = torch.rand(param.shape, generator=gen, dtype=torch.float32) * 3.0 + 0.5
                    param.copy_(torch.log(u))
                elif name.endswith("dt_bias"):
                    param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.2 + 0.5)
                elif "gate_up_projection.weight" in name or "gate_down_projection.weight" in name:
                    param.copy_((torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.15).to(torch.bfloat16).float())
                elif "post_self_attn_layernorm.weight" in name or "post_feedforward_layernorm.weight" in name:
                    w = 0.20 + 0.04 * torch.randn(param.shape, generator=gen, dtype=torch.float32)
                    param.copy_(w.to(torch.bfloat16).float())
                elif name.endswith("shared_head_norm.weight") or name.startswith("model.norm."):
                    w = 0.35 + 0.05 * torch.randn(param.shape, generator=gen, dtype=torch.float32)
                    param.copy_(w.to(torch.bfloat16).float())
                elif name.endswith(".norm.weight") or "layernorm.weight" in name or name.endswith("k_norm.weight") or name.endswith("enorm.weight") or name.endswith("hnorm.weight"):
                    w = 0.85 + 0.10 * torch.randn(param.shape, generator=gen, dtype=torch.float32)
                    param.copy_(w.to(torch.bfloat16).float())
                elif "mlp.gate.weight" in name:
                    param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.35)
                elif "indexer.weights_proj.weight" in name:
                    param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.35)
                elif "embed_tokens.weight" in name:
                    param.copy_((torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.08).to(torch.bfloat16).float())
                else:
                    std = 0.02
                    param.copy_((torch.randn(param.shape, generator=gen, dtype=torch.float32) * std).to(torch.bfloat16).float())
            for name, buf in m.named_buffers():
                if name.endswith("e_score_correction_bias"):
                    buf.copy_(torch.randn(buf.shape, generator=gen, dtype=torch.float32) * 0.05)


def _export_hf_safetensors(hf_dir: Path, tc: dict, model: nn.Module, nextn: _NextNMTPModule) -> None:
    hf_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.endswith("e_score_correction_bias") or k.endswith(("A_log", "dt_bias", "mlp.gate.weight", "indexer.weights_proj.weight")):
            tensors[k] = v.detach().to(torch.float32).contiguous()
        else:
            tensors[k] = v.detach().to(torch.bfloat16).contiguous()

    mblk = f"model.layers.{tc['num_hidden_layers']}."
    for sub_name, mod_attr in (("enorm", nextn.enorm), ("hnorm", nextn.hnorm), ("shared_head.norm", nextn.shared_head_norm)):
        for k, v in mod_attr.state_dict().items():
            tensors[f"{mblk}{sub_name}.{k}"] = v.detach().to(torch.bfloat16).contiguous()
    tensors[f"{mblk}eh_proj.weight"] = nextn.eh_proj.weight.detach().to(torch.bfloat16).contiguous()
    for k, v in nextn.layer.state_dict().items():
        tensors[f"{mblk}{k}"] = v.detach().to(torch.bfloat16).contiguous()

    save_file(tensors, str(hf_dir / "model.safetensors"))
    (hf_dir / "config.json").write_text(json.dumps(tc, indent=2, sort_keys=True))
    (hf_dir / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE"}}')
    (hf_dir / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")


def load_qwp_weights(pack_dir: Path) -> dict[str, torch.Tensor]:
    qp = pack_dir / "model.qwp"
    hdr, _ = read_header(qp)
    out: dict[str, torch.Tensor] = {}
    with qp.open("rb") as fp:
        for name, e in hdr["tensors"].items():
            shape = [int(d) for d in e["shape"]]
            if "raw_off" in e:
                kind = e.get("kind") or e.get("dtype") or "f16"
                bpe = 2 if kind in ("bf16", "f16") else (4 if kind == "f32" else 1)
                n_bytes = int(e["n_bytes"]) if "n_bytes" in e else math.prod(shape) * bpe
                fp.seek(int(e["raw_off"]))
                buf = fp.read(n_bytes)
                if kind == "bf16":
                    arr = torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()).view(torch.bfloat16).float()
                elif kind == "f32":
                    arr = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).reshape(shape).copy())
                else:
                    arr = torch.from_numpy(np.frombuffer(buf, dtype=np.float16).reshape(shape).copy()).float()
            elif "hq4_off" in e:
                hq = e.get("hq")
                bits, grp = (4, 32) if hq is None else (int(hq["bits"]), int(hq["group"]))
                is_m5, is_shvq = (hq is None), bool(hq and hq.get("vq") == "shvq1")
                bbytes = any_block_bytes(bits, grp, is_m5=is_m5, is_shvq=is_shvq)
                k = shape[-1]
                outer = int(np.prod(shape[:-1]))
                fp.seek(int(e["hq4_off"]))
                blocks = np.frombuffer(fp.read(outer * (k // grp) * bbytes), dtype=np.uint8).reshape(outer, k // grp, bbytes).copy()
                arr = torch.from_numpy(dequant_any_blocks(blocks, bits, grp, is_m5=is_m5, is_shvq=is_shvq).reshape(shape))
            else:
                n, k = shape
                fp.seek(int(e["hq8_off"]))
                qs = np.frombuffer(fp.read(n * k), dtype=np.int8).reshape(n, k).copy()
                fp.seek(int(e["hq8_s_off"]))
                sc = np.frombuffer(fp.read(n * (k // 32) * 2), dtype=np.float16).reshape(n, k // 32).copy()
                arr = torch.from_numpy(dequant_hq8_g32_m5(qs, sc))
            out[name] = arr
    return out


def _reinterleave_layer_into_hf(qwp: dict[str, torch.Tensor], tc: dict, blk: str, is_mla: bool) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    h = int(tc["hidden_size"])
    for k, v in qwp.items():
        if not k.startswith(blk):
            continue
        rel = k[len(blk):]
        if rel.startswith("linear_attn.") or rel.endswith(("q_b_nope_proj.weight", "q_b_rope_proj.weight")):
            continue
        sd[rel] = v
    if is_mla:
        qh = int(tc["num_attention_heads"])
        qr = int(tc["q_lora_rank"])
        nope = int(tc["qk_nope_head_dim"])
        rope = int(tc["qk_rope_head_dim"])
        qn = qwp[blk + "self_attn.q_b_nope_proj.weight"].view(qh, nope, qr)
        qro = qwp[blk + "self_attn.q_b_rope_proj.weight"].view(qh, rope, qr)
        sd["self_attn.q_b_proj.weight"] = torch.cat([qn, qro], dim=1).reshape(qh * (nope + rope), qr).contiguous()
    else:
        qk_h = int(tc["linear_num_key_heads"])
        v_h = int(tc["linear_num_value_heads"])
        hd = int(tc["linear_key_head_dim"])
        vp = v_h // qk_h
        qkv = qwp[blk + "linear_attn.in_proj_qkv.weight"]
        z = qwp[blk + "linear_attn.in_proj_z.weight"]
        a = qwp[blk + "linear_attn.in_proj_a.weight"]
        b = qwp[blk + "linear_attn.in_proj_b.weight"]
        q, k_t, v_t = qkv.split([qk_h * hd, qk_h * hd, v_h * hd], dim=0)
        sd["self_attn.in_proj_qkvz.weight"] = torch.cat(
            [q.view(qk_h, hd, h), k_t.view(qk_h, hd, h), v_t.view(qk_h, vp * hd, h), z.view(qk_h, vp * hd, h)],
            dim=1,
        ).reshape(-1, h).contiguous()
        sd["self_attn.in_proj_ba.weight"] = torch.cat(
            [b.view(qk_h, vp, h), a.view(qk_h, vp, h)], dim=1
        ).reshape(-1, h).contiguous()
        for tail in ("conv1d.weight", "norm.weight", "A_log", "dt_bias", "out_proj.weight"):
            sd["self_attn." + tail] = qwp[blk + "linear_attn." + tail]
    return sd


def _load_qwp_into_models(qwp: dict[str, torch.Tensor], tc: dict, model: nn.Module, nextn: _NextNMTPModule) -> None:
    trunk_sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": qwp["model.embed_tokens.weight"],
        "lm_head.weight": qwp["lm_head.weight"],
    }
    for k, v in qwp.items():
        if k.startswith("model.norm."):
            trunk_sd[k] = v
    for l_idx in range(int(tc["num_hidden_layers"])):
        blk = f"model.layers.{l_idx}."
        is_mla = l_idx in tc["full_attention_layers"]
        for rel, val in _reinterleave_layer_into_hf(qwp, tc, blk, is_mla=is_mla).items():
            trunk_sd[blk + rel] = val
    model.load_state_dict(trunk_sd, strict=True)

    mblk = f"model.layers.{tc['num_hidden_layers']}."
    for sub_name, mod_attr in (("enorm.", nextn.enorm), ("hnorm.", nextn.hnorm), ("shared_head.norm.", nextn.shared_head_norm)):
        sub_sd = {k[len(mblk + sub_name):]: v for k, v in qwp.items() if k.startswith(mblk + sub_name)}
        mod_attr.load_state_dict(sub_sd, strict=True)
    nextn.eh_proj.load_state_dict({"weight": qwp[mblk + "eh_proj.weight"]}, strict=True)
    mtp_layer_sd = {
        k: v for k, v in _reinterleave_layer_into_hf(qwp, tc, mblk, is_mla=True).items()
        if not k.startswith(("enorm.", "hnorm.", "shared_head.", "eh_proj."))
    }
    nextn.layer.load_state_dict(mtp_layer_sd, strict=True)


def _softcap_logits(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    if cap is None or cap <= 0:
        return logits.float()
    return torch.tanh(logits.float() / cap) * cap


@torch.no_grad()
def _run_nextn_drafts(
    model: nn.Module,
    nextn: _NextNMTPModule,
    tc: dict,
    prompt_tokens: list[int],
    trunk_hidden: torch.Tensor,
    first_token: int,
    num_drafts: int = 3,
) -> list[dict]:
    t_len = len(prompt_tokens)
    mtp_cache = nextn.cache_cls(config=nextn.mtp_cfg)
    pair_count = t_len - 1
    if pair_count > 0:
        emb_next = model.model.embed_tokens(torch.tensor(prompt_tokens[1:], dtype=torch.long)).unsqueeze(0)
        hid_prev = trunk_hidden[:, :pair_count, :]
        seed_in = nextn.eh_proj(torch.cat([nextn.enorm(emb_next), nextn.hnorm(hid_prev)], dim=-1))
        pos_ids = torch.arange(pair_count, dtype=torch.long).unsqueeze(0)
        cache_pos = torch.arange(pair_count, dtype=torch.long)
        mask_kw = {"inputs_embeds": seed_in} if nextn.uses_inputs_embeds else {"input_embeds": seed_in, "cache_position": cache_pos}
        c_mask = nextn.create_causal_mask(config=nextn.mtp_cfg, attention_mask=None, past_key_values=mtp_cache, position_ids=pos_ids, **mask_kw)
        pos_emb = nextn.rotary_emb(seed_in, pos_ids)
        _ = nextn.layer(seed_in, attention_mask=c_mask, position_ids=pos_ids, past_key_values=mtp_cache, use_cache=True, cache_position=cache_pos, position_embeddings=pos_emb)

    cur_tok = first_token
    cur_hid = trunk_hidden[:, t_len - 1 : t_len, :]
    cap = float(tc["final_logit_softcapping"])
    out_steps: list[dict] = []
    for step in range(num_drafts):
        emb = model.model.embed_tokens(torch.tensor([[cur_tok]], dtype=torch.long))
        x_in = nextn.eh_proj(torch.cat([nextn.enorm(emb), nextn.hnorm(cur_hid)], dim=-1))
        pos_val = pair_count + step
        pos_ids = torch.tensor([[pos_val]], dtype=torch.long)
        cache_pos = torch.tensor([pos_val], dtype=torch.long)
        mask_kw = {"inputs_embeds": x_in} if nextn.uses_inputs_embeds else {"input_embeds": x_in, "cache_position": cache_pos}
        c_mask = nextn.create_causal_mask(config=nextn.mtp_cfg, attention_mask=None, past_key_values=mtp_cache, position_ids=pos_ids, **mask_kw)
        pos_emb = nextn.rotary_emb(x_in, pos_ids)
        block_out = nextn.layer(x_in, attention_mask=c_mask, position_ids=pos_ids, past_key_values=mtp_cache, use_cache=True, cache_position=cache_pos, position_embeddings=pos_emb)
        logits = _softcap_logits(model.lm_head(nextn.shared_head_norm(block_out))[0, 0], cap)
        greedy_tok = int(torch.argmax(logits).item())
        out_steps.append({
            "step": step,
            "token_in": cur_tok,
            "greedy_token": greedy_tok,
            "logits": logits.tolist(),
        })
        cur_tok = greedy_tok
        cur_hid = block_out
    return out_steps


@torch.no_grad()
def generate_oracle_goldens(out_dir: Path, snapshot_dir: Path, seed: int = 20261127) -> dict:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_mod, mod = _load_hf_gigachat4_modules(snapshot_dir)
    _patch_bf16_gated_norm(mod)

    tc = _build_synthetic_config_dict()
    cfg = cfg_mod.GigaChat4Config(**tc)
    cfg._attn_implementation = "eager"
    model = mod.GigaChat4ForCausalLM(cfg).float().eval()
    nextn = _NextNMTPModule(cfg_mod, mod, tc).float().eval()
    _init_deterministic_weights(model, nextn, mod, seed=seed)

    hf_dir = out_dir / "hf_src"
    pack_dir = out_dir / "pack"
    _export_hf_safetensors(hf_dir, tc, model, nextn)
    quantize_gigachat4_model(hf_dir, pack_dir, quant="hq8_g32")

    prompt = [1, 17, 42, 89, 131, 205, 314, 511]
    raw_logits = model(input_ids=torch.tensor([prompt], dtype=torch.long), use_cache=False).logits[0, -1].float()

    qwp = load_qwp_weights(pack_dir)
    _load_qwp_into_models(qwp, tc, model, nextn)

    captured: list[torch.Tensor] = []
    hook = model.model.layers[-1].register_forward_hook(lambda _m, _i, o: captured.append(o.detach().clone()))
    cache = mod.GigaChat4DynamicCache(config=cfg)
    pre_out = model(input_ids=torch.tensor([prompt], dtype=torch.long), past_key_values=cache, use_cache=True)
    hook.remove()

    trunk_hidden = captured[0]
    pre_logits = pre_out.logits[0, -1].float()
    pre_tok = int(torch.argmax(pre_logits).item())
    hf_vs_hq8_max_diff = float((raw_logits - pre_logits).abs().max().item())

    nextn_drafts = _run_nextn_drafts(model, nextn, tc, prompt, trunk_hidden, pre_tok, num_drafts=3)

    decode_steps: list[dict] = []
    greedy_seq: list[int] = [pre_tok]
    cur = pre_tok
    for step in range(7):
        step_out = model(input_ids=torch.tensor([[cur]], dtype=torch.long), past_key_values=cache, use_cache=True)
        lg = step_out.logits[0, -1].float()
        nxt = int(torch.argmax(lg).item())
        if step < 4:
            decode_steps.append({"step": step, "token_in": cur, "greedy_token": nxt, "logits": lg.tolist()})
        greedy_seq.append(nxt)
        cur = nxt

    batch_prompts = [
        [1, 17, 42, 89, 131, 205, 314, 511],
        [1, 9, 33, 77, 144, 288],
        [1, 51, 62, 73, 84, 95, 106],
        [1, 210, 310, 410, 510],
    ]
    b_caches = [mod.GigaChat4DynamicCache(config=cfg) for _ in batch_prompts]
    b_cur: list[int] = []
    for b_idx, bp in enumerate(batch_prompts):
        bo = model(input_ids=torch.tensor([bp], dtype=torch.long), past_key_values=b_caches[b_idx], use_cache=True)
        b_cur.append(int(torch.argmax(bo.logits[0, -1].float()).item()))
    batch_prefill_tokens = list(b_cur)

    batch_decode_steps: list[dict] = []
    for step in range(2):
        step_logits: list[list[float]] = []
        step_next: list[int] = []
        for b_idx in range(len(batch_prompts)):
            bo = model(
                input_ids=torch.tensor([[b_cur[b_idx]]], dtype=torch.long),
                past_key_values=b_caches[b_idx],
                use_cache=True,
            )
            lg = bo.logits[0, -1].float()
            nxt = int(torch.argmax(lg).item())
            step_logits.append(lg.tolist())
            step_next.append(nxt)
        batch_decode_steps.append({"step": step, "tokens_in": list(b_cur), "greedy_tokens": step_next, "logits": step_logits})
        b_cur = step_next

    report = {
        "pack_dir": str(pack_dir),
        "vocab_size": int(tc["vocab_size"]),
        "hidden_size": int(tc["hidden_size"]),
        "hf_vs_hq8_prefill_max_diff": hf_vs_hq8_max_diff,
        "prompt_tokens": prompt,
        "prefill_greedy_token": pre_tok,
        "prefill_last_logits": pre_logits.tolist(),
        "decode_steps": decode_steps,
        "greedy_sequence_8": greedy_seq,
        "batch_prompts": batch_prompts,
        "batch_prefill_tokens": batch_prefill_tokens,
        "batch_decode_steps": batch_decode_steps,
        "nextn_drafts": nextn_drafts,
    }
    (out_dir / "oracle_goldens.json").write_text(json.dumps(report))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--hf-snapshot",
        type=Path,
        default=Path(os.environ.get("QW_GIGACHAT4_HF_SNAPSHOT", str(DEFAULT_HF_SNAPSHOT))),
    )
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args(argv)
    rep = generate_oracle_goldens(args.out_dir, args.hf_snapshot, seed=args.seed)
    print(
        json.dumps(
            {
                "pack_dir": rep["pack_dir"],
                "hf_vs_hq8_prefill_max_diff": rep["hf_vs_hq8_prefill_max_diff"],
                "prefill_greedy_token": rep["prefill_greedy_token"],
                "greedy_sequence_8": rep["greedy_sequence_8"],
                "nextn_draft_tokens": [d["greedy_token"] for d in rep["nextn_drafts"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
