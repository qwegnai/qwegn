#!/usr/bin/env python3
"""Offline AQ8_G32 KL + Top-1 oracle for Inkling-2-Mini router gates & loop controllers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

QUANTIZE_DIR = Path(__file__).resolve().parents[1] / "quantize"
if str(QUANTIZE_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTIZE_DIR))

from hq_blocks import dequant_hq_blocks_rot, quant_hq_blocks  # noqa: E402
from inkling2_adapter import (  # noqa: E402
    Inkling2Source,
    Inkling2Spec,
    dequant_bf16,
    dequant_mxfp8,
    dequant_nvfp4,
    maybe_force_cpu_quant,
    open_inkling2_shards,
    parse_inkling2_config,
)
from pack import resolve_snapshot_dir  # noqa: E402
from score_teacher_logits import score as score_kl  # noqa: E402

DEFAULT_MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"
DIVERSE_CALIBRATION_TEXT = (
    "def topk_router_parity(logits: list[float], bias: list[float], k: int = 8) -> list[int]:\n"
    "    scores = [1.0 / (1.0 + math.exp(-v)) for v in logits]\n"
    "    ranked = sorted(range(len(scores)), key=lambda i: (-(scores[i] + bias[i]), i))\n"
    "    return ranked[:k]\n\n"
    "Theorem (Parseval & Jensen): For any orthonormal H_32, ||H_32 x||_2 = ||x||_2 and "
    "KL(P || Q) = sum_i P_i * log(P_i / Q_i) >= 0 with equality iff P == Q.\n\n"
    '{"tool": "run_eval", "args": {"context_tokens": 131072, "kv_mode": "q4", '
    '"speculative": true, "thresholds": {"delta_kl": 0.002, "top1_flip_pct": 0.5}}}\n\n'
    "Across 48 recurrent visits in an Inkling-2 transformer, message banks accumulate "
    "normalized residual states while low-rank forget/gain controllers gate cross-pass updates "
    "before sparse top-k experts project activations onto unpadded vocabulary logits."
)

VARIANTS = ("router_aq8_g32", "ctrl_aq8_g32", "router_ctrl_aq8_g32")


def quant_dequant_aq8_g32(w: torch.Tensor) -> torch.Tensor:
    """Quantize 2D tensor to AQ8_G32 (rotate=False) and reconstruct float32."""
    with maybe_force_cpu_quant("cpu"):
        blks, eff_g = quant_hq_blocks(w.to(torch.float32), bits=8, group=32, is_m5=False, rotate=False, mse_clip=False)
    recon = dequant_hq_blocks_rot(blks, bits=8, group=eff_g, is_m5=False)
    return torch.from_numpy(recon).to(torch.float32)


def tensor_err(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    r64, c64 = ref.to(torch.float64), cand.to(torch.float64)
    diff = r64 - c64
    return {
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "rel_l2": float(torch.linalg.norm(diff) / (torch.linalg.norm(r64) + 1e-12)),
        "cosine": float(F.cosine_similarity(r64.reshape(-1), c64.reshape(-1), dim=0)),
        "max_abs": float(diff.abs().max()),
    }


def inspect_runtime_packs(models_dir: Path | None) -> dict[str, dict]:
    """Read QWP1 JSON headers to verify routerGate and controller storage formats."""
    if models_dir is None or not models_dir.is_dir():
        return {}
    out: dict[str, dict] = {}
    for name in ("Inkling-2-Mini-QWP-Fast", "Inkling-2-Mini-QWP-Cand5"):
        qwp = models_dir / name / "model.qwp"
        if not qwp.is_file():
            continue
        with qwp.open("rb") as f:
            if f.read(4) != b"QWP1":
                continue
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(-hdr_len, 2)
            hdr = json.loads(f.read(hdr_len))
        tensors = hdr.get("tensors", {})
        router_kinds, ctrl_kinds = set(), set()
        for tname, entry in tensors.items():
            kind = ("raw_" + entry.get("kind", "bf16")) if "raw_off" in entry else (
                f"{(entry.get('hq') or {}).get('vq', 'hq4')}_b{(entry.get('hq') or {}).get('bits', 4)}_g{(entry.get('hq') or {}).get('group', 32)}"
            )
            if tname.endswith("mlp.gate_weight"):
                router_kinds.add(kind)
            elif ".controllers." in tname:
                ctrl_kinds.add(kind)
        out[name] = {
            "quant_recorded": (hdr.get("provenance") or {}).get("quant_recorded"),
            "router_gate_disk": sorted(router_kinds),
            "controllers_disk": sorted(ctrl_kinds),
            "controllers_runtime": "bf16 (dequantized at load if quantized on disk)",
        }
    return out


def byte_budget_summary(spec: Inkling2Spec) -> dict[str, float]:
    h, r, e = spec.hidden_size, spec.loop_gate_rank, spec.num_experts
    core_l = spec.loop_core_end - spec.loop_core_start
    core_visits = core_l * max(0, spec.loop_num_passes - 1)
    r_bf16 = e * h * 2
    r_aq8 = e * (h // 32) * 36
    c_bf16 = 4 * (r * 2 * h + 2 * h * r) * 2
    c_aq8 = 4 * (r * (2 * h // 32) + 2 * h * (r // 32)) * 36
    return {
        "router_stored_bf16_mib": (spec.num_layers * r_bf16) / float(1 << 20),
        "router_stored_aq8_mib": (spec.num_layers * r_aq8) / float(1 << 20),
        "router_active_pass_bf16_mib": (len(spec.execution_plan) * r_bf16) / float(1 << 20),
        "router_active_pass_aq8_mib": (len(spec.execution_plan) * r_aq8) / float(1 << 20),
        "ctrl_stored_bf16_mib": (core_l * c_bf16) / float(1 << 20),
        "ctrl_stored_aq8_mib": (core_l * c_aq8) / float(1 << 20),
        "ctrl_active_pass_bf16_mib": (core_visits * c_bf16) / float(1 << 20),
        "ctrl_active_pass_aq8_mib": (core_visits * c_aq8) / float(1 << 20),
    }


def build_diverse_tokens(model_dir: Path, unpadded_vocab: int, num_tokens: int) -> np.ndarray:
    tok_path = model_dir / "tokenizer.json"
    if tok_path.is_file():
        try:
            from tokenizers import Tokenizer
            ids = [i for i in Tokenizer.from_file(str(tok_path)).encode(DIVERSE_CALIBRATION_TEXT).ids if 0 <= i < unpadded_vocab]
            if ids:
                while len(ids) < num_tokens:
                    ids.extend(ids)
                return np.asarray(ids[:num_tokens], dtype=np.int64)
        except Exception:
            pass
    return np.random.default_rng(20261123).integers(0, unpadded_vocab, size=(num_tokens,), dtype=np.int64)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w


def _tanh_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(torch.clamp(x, -15.0, 15.0))


def _sconv_static(x: torch.Tensor, w: torch.Tensor, denom: float = 5.0) -> torch.Tensor:
    xp = F.pad(x, (0, 0, 3, 0))
    acc = xp[:-3] * w[:, 0] + xp[1:-2] * w[:, 1] + xp[2:-1] * w[:, 2] + x * w[:, 3]
    return (acc + x) / math.sqrt(denom)


def _sconv_dynamic(x: torch.Tensor, st_w: torch.Tensor, dyn_w: torch.Tensor, n_kv: int, head_dim: int) -> torch.Tensor:
    t_len, kv_dim = x.shape
    dw = dyn_w.reshape(t_len, n_kv, 4).unsqueeze(2).expand(t_len, n_kv, head_dim, 4).reshape(t_len, kv_dim, 4)
    w = st_w.t().unsqueeze(0) + dw
    xp = F.pad(x, (0, 0, 3, 0))
    return (w[..., 0] * x + w[..., 1] * xp[2:-1] + w[..., 2] * xp[1:-2] + w[..., 3] * xp[:-3] + x) / 3.0


def _attn_res(blocks: list[torch.Tensor], partial: torch.Tensor, queries: torch.Tensor, norm_val: bool) -> torch.Tensor:
    cands = torch.stack(blocks + [partial], dim=0)
    h = partial.shape[-1]
    inv_rms = torch.rsqrt(cands.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    alpha = torch.einsum("cth,qh->qtc", cands, queries.reshape(-1, h)) * inv_rms.squeeze(-1).t().unsqueeze(0) / math.sqrt(h)
    vals = (cands * inv_rms) if norm_val else cands
    return torch.einsum("qtc,cth->qth", torch.softmax(alpha, dim=-1), vals)


def _msg_controller(x: torch.Tensor, msg: torch.Tensor, dw: torch.Tensor, fw: torch.Tensor, gw: torch.Tensor, eps: float) -> torch.Tensor:
    inv_x = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    inv_m = torch.rsqrt(msg.pow(2).mean(dim=-1, keepdim=True) + eps)
    hid = torch.cat([x * inv_x, msg * inv_m], dim=-1) @ dw.t()
    return (1.0 - _tanh_safe(hid @ fw.t())) * x + _tanh_safe(hid @ gw.t()) * msg


def _load_active_experts(src: Inkling2Source, layer: int, proj: str, active: list[int]) -> dict[int, torch.Tensor]:
    if not active:
        return {}
    mode = src.spec.expert_mode(layer, proj)
    idx_t = torch.tensor(active, dtype=torch.long)
    if src.layout == "mlx":
        sw = f"{src.in_pfx}model.layers.{layer}.mlp.switch_mlp.{proj}."
        w = src.reader.get(sw + "weight")[idx_t]
        if mode == "nvfp4":
            deq = dequant_nvfp4(w, src.reader.get(sw + "scales")[idx_t], src.reader.get(sw + "weight_scale_2")[idx_t], as_torch=True)
        elif mode == "mxfp8":
            deq = dequant_mxfp8(w, src.reader.get(sw + "scales")[idx_t], as_torch=True)
        else:
            deq = dequant_bf16(w, as_torch=True)
    else:
        ep, stem = f"model.llm.layers.{layer}.mlp.experts.", ("w2_weight" if proj == "down_proj" else "w13_weight")
        row_sl = slice(0, None, 2) if proj == "gate_proj" else (slice(1, None, 2) if proj == "up_proj" else slice(None))
        w = src.reader.get(ep + stem)[idx_t, row_sl].contiguous()
        if mode == "nvfp4":
            sc = src.reader.get(ep + stem + "_scale")[idx_t, row_sl].contiguous()
            s2 = (src.reader.get(ep + stem + "_amax")[idx_t].float() / src.reader.get(ep + stem + "_outer_scale_denominator").float()).contiguous()
            deq = dequant_nvfp4(w, sc, s2, as_torch=True)
        elif mode == "mxfp8":
            sc = src.reader.get(ep + stem + "_scale")[idx_t, row_sl].contiguous()
            deq = dequant_mxfp8(w, sc, as_torch=True)
        else:
            deq = dequant_bf16(w, as_torch=True)
    return {e: deq[i] for i, e in enumerate(active)}


def _load_layer_shared(src: Inkling2Source, layer: int, err_acc: dict[str, list[dict[str, float]]]) -> dict:
    spec = src.spec
    is_sl = spec.is_sliding(layer)
    rg_bf16 = dequant_bf16(src.get_layer(layer, "mlp.gate_weight"))
    rg_aq8 = quant_dequant_aq8_g32(rg_bf16)
    err_acc["router_gate"].append(tensor_err(rg_bf16, rg_aq8))
    d: dict = {
        "mlp_input_query": src.get_layer(layer, "mlp_input.query").to(torch.float32),
        "post_attn_norm": dequant_bf16(src.get_layer(layer, "post_attention_layernorm.weight")),
        "attn_sconv": dequant_bf16(src.get_layer(layer, "attn_sconv.conv.weight")).reshape(-1, 4),
        "mlp_sconv": dequant_bf16(src.get_layer(layer, "mlp_sconv.conv.weight")).reshape(-1, 4),
        "q_scale": dequant_bf16(src.get_layer(layer, "self_attn.q_scale")),
        "rel_scale": dequant_bf16(src.get_layer(layer, "self_attn.rel_scale")),
        "rel_proj": dequant_bf16(src.get_layer(layer, "self_attn.rel_proj")),
        "router_gate_bf16": rg_bf16, "router_gate_aq8": rg_aq8,
        "router_bias": src.get_layer(layer, "mlp.e_score_correction_bias").to(torch.float32),
    }
    if layer > 0:
        d["attn_input_query"] = src.get_layer(layer, "attention_input.query").to(torch.float32)
    if is_sl:
        d["input_norm"] = dequant_bf16(src.get_layer(layer, "input_layernorm.weight"))
        d["k_sconv"] = dequant_bf16(src.get_layer(layer, "self_attn.k_sconv.conv.weight")).reshape(-1, 4)
        d["v_sconv"] = dequant_bf16(src.get_layer(layer, "self_attn.v_sconv.conv.weight")).reshape(-1, 4)
    else:
        for k in ("q", "k", "v"):
            d[f"{k}_norm"] = dequant_bf16(src.get_layer(layer, f"{k}_input_layernorm.weight"))
        d["k_sconv_static"] = dequant_bf16(src.get_layer(layer, "self_attn.k_sconv.static_weight"))
        d["v_sconv_static"] = dequant_bf16(src.get_layer(layer, "self_attn.v_sconv.static_weight"))
        d["k_sconv_proj"] = dequant_bf16(src.get_layer(layer, "self_attn.k_sconv.weight_proj.weight"))
        d["v_sconv_proj"] = dequant_bf16(src.get_layer(layer, "self_attn.v_sconv.weight_proj.weight"))
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "r_proj", "sink_proj"):
        d[p] = dequant_bf16(src.get_layer(layer, f"self_attn.{p}.weight"))
    if src.has_layer(layer, "self_attn.q_proj.bias"):
        d["q_bias"] = dequant_bf16(src.get_layer(layer, "self_attn.q_proj.bias"))
    if src.has_layer(layer, "self_attn.o_proj.bias"):
        d["o_bias"] = dequant_bf16(src.get_layer(layer, "self_attn.o_proj.bias"))
    if spec.loop_core_start <= layer < spec.loop_core_end:
        c_bf16, c_aq8 = {}, {}
        for cn in ("attn", "k", "v", "mlp"):
            dw = dequant_bf16(src.get_layer(layer, f"controllers.{cn}.down_weight"))
            fw = dequant_bf16(src.get_layer(layer, f"controllers.{cn}.forget_weight"))
            gw = dequant_bf16(src.get_layer(layer, f"controllers.{cn}.gain_weight"))
            dw_q, fw_q, gw_q = quant_dequant_aq8_g32(dw), quant_dequant_aq8_g32(fw), quant_dequant_aq8_g32(gw)
            err_acc["ctrl_down"].append(tensor_err(dw, dw_q))
            err_acc["ctrl_forget"].append(tensor_err(fw, fw_q))
            err_acc["ctrl_gain"].append(tensor_err(gw, gw_q))
            c_bf16[cn], c_aq8[cn] = (dw, fw, gw), (dw_q, fw_q, gw_q)
        d["ctrl_bf16"], d["ctrl_aq8"] = c_bf16, c_aq8
    return d


def eval_variants(src: Inkling2Source, token_ids: torch.Tensor) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Run teacher + 3 AQ8_G32 variants simultaneously, loading each layer once."""
    spec = src.spec
    all_names = ("teacher_bf16", *VARIANTS)
    with torch.no_grad():
        emb_tbl = dequant_bf16(src.get_global("embed_tokens.weight"))
        emb_norm = dequant_bf16(src.get_global("embed_norm.weight"))
        final_norm = dequant_bf16(src.get_global("norm.weight"))
        final_query = src.get_global("final_input.query").to(torch.float32)
        head_w = dequant_bf16(src.get_global("lm_head.weight"))[: spec.unpadded_vocab_size]
        core_len = spec.loop_core_end - spec.loop_core_start
        msg_q = src.get_global("message_queries").to(torch.float32).reshape(core_len, 4, spec.hidden_size)

        t_len, h = int(token_ids.shape[0]), spec.hidden_size
        n_h, n_kv, d, d_rel, eps = spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim, spec.d_rel, spec.rms_norm_eps
        rep = n_h // n_kv
        t_idx = torch.arange(t_len)[:, None]
        dist_mat = t_idx - torch.arange(t_len)[None, :]
        causal_mask = torch.triu(torch.full((t_len, t_len), float("-inf")), diagonal=1)
        pos_idx = torch.arange(t_len, dtype=torch.float32)

        x0 = _rms_norm(emb_tbl[token_ids], emb_norm, eps)
        states = {v: {"x": x0.clone(), "blocks": [], "prelude": None, "pblocks": 0, "bank": None, "bank_ok": False} for v in all_names}
        err_acc: dict[str, list[dict[str, float]]] = {"router_gate": [], "ctrl_down": [], "ctrl_forget": [], "ctrl_gain": []}
        route_stats = {v: {"top1_match": 0, "topk_jacc_sum": 0.0, "topk_exact": 0, "total_pos": 0} for v in VARIANTS}
        layer_cache: dict[int, dict] = {}
        expert_cache: dict[int, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]] = {}

        for visit, l_idx in enumerate(spec.execution_plan):
            t0 = time.time()
            if l_idx not in layer_cache:
                lw = _load_layer_shared(src, l_idx, err_acc)
                if spec.loop_core_start <= l_idx < spec.loop_core_end:
                    layer_cache[l_idx] = lw
            else:
                lw = layer_cache[l_idx]
            is_sl = spec.is_sliding(l_idx)
            max_dist = min(spec.rel_extent, t_len)
            mask = causal_mask.masked_fill(dist_mat >= spec.sliding_window_size, float("-inf")) if (is_sl and spec.sliding_window_size > 0) else causal_mask

            pre_moe: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            needed_experts: set[int] = set()
            for v in all_names:
                st = states[v]
                if l_idx == spec.loop_core_start:
                    if visit == l_idx:
                        st["prelude"], st["pblocks"] = st["x"].clone(), len(st["blocks"])
                    else:
                        b_states = torch.stack(st["blocks"][st["pblocks"] + 1 :] + [st["x"]], dim=0)
                        st["bank"] = b_states * torch.rsqrt(b_states.pow(2).mean(dim=-1, keepdim=True) + eps)
                        st["bank_ok"] = True
                        st["x"], st["blocks"] = st["prelude"].clone(), st["blocks"][: st["pblocks"]]
                use_bank = st["bank_ok"] and (spec.loop_core_start <= l_idx < spec.loop_core_end)
                msg_out = None
                if use_bank:
                    dots = torch.einsum("mth,qh->qtm", st["bank"], msg_q[l_idx - spec.loop_core_start]) / math.sqrt(h)
                    msg_out = torch.einsum("qtm,mth->qth", torch.softmax(dots, dim=-1), st["bank"])

                use_ctrl_aq8 = v in ("ctrl_aq8_g32", "router_ctrl_aq8_g32")
                ctrls = lw.get("ctrl_aq8" if use_ctrl_aq8 else "ctrl_bf16")
                if l_idx == 0 and is_sl:
                    q_in = k_in = v_in = _rms_norm(st["x"], lw["input_norm"], eps)
                else:
                    ar_out = _attn_res(st["blocks"], st["x"], lw["attn_input_query"], spec.attn_res_normalize_values)
                    if is_sl and not use_bank:
                        q_in = k_in = v_in = _rms_norm(ar_out[0], lw["input_norm"], eps)
                    else:
                        nw = ([lw["input_norm"]] * 3) if is_sl else [lw["q_norm"], lw["k_norm"], lw["v_norm"]]
                        normed = []
                        for i, cn in enumerate(("attn", "k", "v")):
                            ar_i = ar_out[min(i, ar_out.shape[0] - 1)]
                            if use_bank and ctrls is not None and msg_out is not None:
                                ar_i = _msg_controller(ar_i, msg_out[i], *ctrls[cn], eps)
                            normed.append(_rms_norm(ar_i, nw[i], eps))
                        q_in, k_in, v_in = normed

                q_proj = q_in @ lw["q_proj"].t() + (lw["q_bias"] if "q_bias" in lw else 0.0)
                r_proj, sink_proj, gate_proj = q_in @ lw["r_proj"].t(), q_in @ lw["sink_proj"].t(), q_in @ lw["gate_proj"].t()
                k_proj, v_proj = k_in @ lw["k_proj"].t(), v_in @ lw["v_proj"].t()
                if is_sl:
                    k_conv, v_conv = _sconv_static(k_proj, lw["k_sconv"]), _sconv_static(v_proj, lw["v_sconv"])
                else:
                    k_conv = _sconv_dynamic(k_proj, lw["k_sconv_static"], k_in @ lw["k_sconv_proj"].t(), n_kv, d)
                    v_conv = _sconv_dynamic(v_proj, lw["v_sconv_static"], v_in @ lw["v_sconv_proj"].t(), n_kv, d)

                qv = q_proj.reshape(t_len, n_h, d) * lw["q_scale"].reshape(1, n_h, d)
                q_prep = qv * torch.clamp(spec.qk_l2_clip_threshold / (qv.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
                rv = r_proj.reshape(t_len, n_h, d_rel) * lw["rel_scale"].reshape(1, n_h, d_rel)
                r_prep = rv * torch.clamp(spec.rel_l2_clip_threshold / (rv.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
                if (not is_sl) and spec.log_scaling_n_floor > 0:
                    ratio = (pos_idx + 1.0) / float(spec.log_scaling_n_floor)
                    tau = torch.where(ratio > 1.0, 1.0 + spec.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)), 1.0).reshape(t_len, 1, 1)
                    q_prep, r_prep = q_prep * tau, r_prep * tau

                kv_r, vv_r = k_conv.reshape(t_len, n_kv, d), v_conv.reshape(t_len, n_kv, d)
                k_prep = kv_r * torch.clamp(spec.qk_l2_clip_threshold / (kv_r.norm(dim=-1, keepdim=True) + 1e-6), max=1.0)
                v_prep = vv_r * torch.rsqrt(vv_r.pow(2).mean(dim=-1, keepdim=True) + eps)

                scores = torch.einsum("thd,shd->hts", q_prep, k_prep.repeat_interleave(rep, dim=1)) / math.sqrt(d)
                rel_bias = torch.einsum("thd,dr->htr", r_prep, lw["rel_proj"][:, :max_dist])[:, t_idx, torch.clamp(dist_mat, 0, max_dist - 1)]
                scores = scores + torch.where(((dist_mat >= 0) & (dist_mat < max_dist)).unsqueeze(0), rel_bias, 0.0)
                probs = torch.softmax(torch.cat([scores + mask.unsqueeze(0), sink_proj.t().unsqueeze(-1)], dim=-1), dim=-1)[..., :t_len]
                o_raw = torch.einsum("hts,shd->thd", probs, v_prep.repeat_interleave(rep, dim=1)).reshape(t_len, n_h * d) * torch.sigmoid(gate_proj)
                o_out = o_raw @ lw["o_proj"].t() + (lw["o_bias"] if "o_bias" in lw else 0.0)

                if spec.is_boundary(l_idx):
                    st["blocks"].append(st["x"].clone())
                    res_f32 = _sconv_static(o_out, lw["attn_sconv"])
                else:
                    res_f32 = _sconv_static(o_out, lw["attn_sconv"]) + st["x"]
                mlp_ar = _attn_res(st["blocks"], res_f32, lw["mlp_input_query"], spec.attn_res_normalize_values)[0]
                if use_bank and ctrls is not None and msg_out is not None:
                    mlp_ar = _msg_controller(mlp_ar, msg_out[3], *ctrls["mlp"], eps)
                mlp_in = _rms_norm(mlp_ar, lw["post_attn_norm"], eps)

                rg_w = lw["router_gate_aq8" if v in ("router_aq8_g32", "router_ctrl_aq8_g32") else "router_gate_bf16"]
                r_scores = torch.sigmoid(mlp_in @ rg_w.t())
                biased = r_scores + lw["router_bias"].unsqueeze(0) if spec.use_gate_bias else r_scores
                _, top_idx = torch.topk(biased, k=spec.experts_per_tok, dim=-1)
                top_w = torch.gather(r_scores, 1, top_idx)
                top_w = top_w * (spec.route_scale / (top_w.sum(dim=-1, keepdim=True) + 1e-20))
                pre_moe[v] = (res_f32, mlp_in, top_idx, top_w)
                needed_experts.update(top_idx.reshape(-1).tolist())

            t_top = pre_moe["teacher_bf16"][2].tolist()
            for v in VARIANTS:
                v_top = pre_moe[v][2].tolist()
                rs = route_stats[v]
                for row_t, row_v in zip(t_top, v_top):
                    rs["top1_match"] += int(row_t[0] == row_v[0])
                    st_set, sv_set = set(row_t), set(row_v)
                    rs["topk_exact"] += int(st_set == sv_set)
                    rs["topk_jacc_sum"] += len(st_set & sv_set) / float(len(st_set | sv_set))
                    rs["total_pos"] += 1

            cached_g, cached_u, cached_d = expert_cache.get(l_idx, ({}, {}, {}))
            missing = sorted(e for e in needed_experts if e not in cached_g)
            if missing:
                cached_g.update(_load_active_experts(src, l_idx, "gate_proj", missing))
                cached_u.update(_load_active_experts(src, l_idx, "up_proj", missing))
                cached_d.update(_load_active_experts(src, l_idx, "down_proj", missing))
                if spec.loop_core_start <= l_idx < spec.loop_core_end:
                    expert_cache[l_idx] = (cached_g, cached_u, cached_d)

            for v in all_names:
                res_f32, mlp_in, top_idx, top_w = pre_moe[v]
                moe_out = torch.zeros((t_len, h), dtype=torch.float32)
                for e in sorted(set(top_idx.reshape(-1).tolist())):
                    tok_i, slot_i = torch.where(top_idx == e)
                    xe = mlp_in[tok_i]
                    ge, ue = xe @ cached_g[e].t(), xe @ cached_u[e].t()
                    act = (_tanh_safe(ge / spec.mlp_gate_softcap) * spec.mlp_gate_softcap * torch.sigmoid(ge)) * (_tanh_safe(ue / spec.mlp_up_softcap) * spec.mlp_up_softcap)
                    moe_out.index_add_(0, tok_i, top_w[tok_i, slot_i].unsqueeze(-1) * (act @ cached_d[e].t()))
                states[v]["x"] = res_f32 + _sconv_static(moe_out, lw["mlp_sconv"])
            print(f"  visit {visit + 1:2d}/{len(spec.execution_plan)} (layer {l_idx:2d}): experts={len(needed_experts)} dt={time.time() - t0:.2f}s", flush=True)

        logits_out: dict[str, np.ndarray] = {}
        for v in all_names:
            st = states[v]
            final_h = _rms_norm(_attn_res(st["blocks"], st["x"], final_query, spec.attn_res_normalize_values)[0], final_norm, eps)
            logits = final_h @ head_w.t()
            if spec.final_logit_softcapping > 0:
                logits = _tanh_safe(logits / spec.final_logit_softcapping) * spec.final_logit_softcapping
            logits_out[v] = logits.cpu().numpy().astype(np.float32)

    err_summary = {
        k: {m: float(np.mean([row[m] for row in rows])) for m in ("rmse", "rel_l2", "cosine", "max_abs")} if rows else {}
        for k, rows in err_acc.items()
    }
    return logits_out, route_stats, err_summary


def run_eval(model_dir: Path | None, *, dry_run: bool, num_tokens: int, models_dir: Path | None) -> dict:
    with tempfile.TemporaryDirectory(prefix="ink2_aq8_kl_") as tmp:
        if dry_run:
            from eval_ink2_quant_kl import make_synthetic_inkling2_checkpoint  # noqa: E402
            src_dir = make_synthetic_inkling2_checkpoint(Path(tmp) / "synth", layout="hf")
        else:
            assert model_dir is not None
            src_dir = resolve_snapshot_dir(model_dir)
        cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
        reader, _ = open_inkling2_shards(src_dir)
        try:
            spec = parse_inkling2_config(cfg, source_dir=src_dir, reader=reader)
            src = Inkling2Source(reader, spec)
            tok_np = build_diverse_tokens(src_dir, spec.unpadded_vocab_size, num_tokens)
            contexts = np.full((num_tokens, num_tokens), -1, dtype=np.int64)
            for p in range(num_tokens):
                contexts[p, : p + 1] = tok_np[: p + 1]
            logits, route_stats, err_summary = eval_variants(src, torch.from_numpy(tok_np))
        finally:
            reader.close()

    teacher_logits = logits["teacher_bf16"]
    variant_reports: dict[str, dict] = {}
    for v in VARIANTS:
        sc = score_kl(teacher_logits, logits[v], contexts, contexts)
        rs = route_stats[v]
        tot = max(1, rs["total_pos"])
        flip_pct = (1.0 - sc["top1_agreement"]) * 100.0
        variant_reports[v] = {
            "mean_kl": sc["mean_kl"],
            "p95_kl": sc["p95_kl"],
            "max_kl": sc["max_kl"],
            "top1_agreement_pct": sc["top1_agreement"] * 100.0,
            "top1_flip_pct": flip_pct,
            "router_top1_match_pct": (rs["top1_match"] / tot) * 100.0,
            "router_topk_exact_pct": (rs["topk_exact"] / tot) * 100.0,
            "router_topk_jaccard": rs["topk_jacc_sum"] / tot,
            "pass_kl_gate": bool(sc["mean_kl"] <= 0.002 and flip_pct <= 0.5),
        }

    return {
        "dry_run": dry_run,
        "source_dir": str(src_dir),
        "num_tokens": num_tokens,
        "visits": len(spec.execution_plan),
        "packs": inspect_runtime_packs(models_dir),
        "byte_budget": byte_budget_summary(spec),
        "per_tensor_aq8_error": err_summary,
        "variants": variant_reports,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--models-dir", type=Path, default=Path("~/.qwg/models").expanduser())
    ap.add_argument("--num-tokens", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.dry_run and not args.model.exists():
        ap.error(f"model directory not found: {args.model}")
    report = run_eval(None if args.dry_run else args.model, dry_run=args.dry_run, num_tokens=args.num_tokens, models_dir=args.models_dir)
    print("\n=== Inkling-2 AQ8_G32 Offline KL Oracle ===", flush=True)
    print("Packs:", json.dumps(report["packs"], indent=2))
    print("Byte budget:", json.dumps(report["byte_budget"], indent=2))
    print("Per-tensor AQ8 error:", json.dumps(report["per_tensor_aq8_error"], indent=2))
    hdr = f"{'Variant':22s} | {'Mean KL':>9s} | {'P95 KL':>9s} | {'Top1 Flip':>9s} | {'RtrExact':>8s} | {'Gate'}"
    print(hdr)
    print("-" * len(hdr))
    for v, r in report["variants"].items():
        gate = "PASS" if r["pass_kl_gate"] else "FAIL"
        print(f"{v:22s} | {r['mean_kl']:9.6f} | {r['p95_kl']:9.6f} | {r['top1_flip_pct']:8.3f}% | {r['router_topk_exact_pct']:7.2f}% | {gate}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
