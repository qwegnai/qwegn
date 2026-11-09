#!/usr/bin/env python3
"""2-Stage KLD / MSE Fidelity & Memory Budget Harness for GLM-5.3-Flash on Qwegn.

Stage 1: Per-subsystem and per-projection relative Frobenius/activation error and cosine
         similarity across HQ{2,3,4,6}_G{32,64,128,256} (with and without mse_clip), plus
         empirical router hit distribution across all 288 experts using e_score_correction_bias.
Stage 2: End-to-end multi-layer (4-stream Sinkhorn-20 mHC + Dense SwiGLU + KDA Linear Attn +
         NoPE MLA + 288-Expert MoE + Final Norm + 154,880-vocab LM Head) teacher vs student
         logit KLD and Top-1 agreement gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hq_blocks import (
    dequant_fp8_blockwise,
    dequant_hq_blocks,
    hq_block_bytes,
    parse_quant_spec,
    quant_hq_blocks,
)


def build_tensor_locator(snap_dir: Path) -> dict[str, Path]:
    locator: dict[str, Path] = {}
    for p in sorted(snap_dir.glob("*.safetensors")):
        with safe_open(p, framework="pt") as sf:
            for k in sf.keys():
                locator[k] = p
    return locator


def load_tensor_f32(locator: dict[str, Path], name: str, device: torch.device) -> torch.Tensor:
    p = locator[name]
    with safe_open(p, framework="pt") as sf:
        t = sf.get_tensor(name)
    scale_name = name + "_scale_inv"
    if scale_name in locator:
        ps = locator[scale_name]
        with safe_open(ps, framework="pt") as sf:
            s = sf.get_tensor(scale_name)
        return dequant_fp8_blockwise(t, s, device=device)
    return t.to(torch.float32).to(device=device)


def quant_dequant_torch(w: torch.Tensor, spec: str, mse_clip: bool | None = None) -> torch.Tensor:
    bits, group, is_m5 = parse_quant_spec(spec)
    blks, eff_g = quant_hq_blocks(w, bits=bits, group=group, is_m5=is_m5, mse_clip=mse_clip)
    dq = dequant_hq_blocks(blks, bits=bits, group=eff_g, is_m5=is_m5)
    return torch.from_numpy(dq).to(device=w.device, dtype=torch.float32)


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    g = torch.clamp(gate, max=limit)
    u = torch.clamp(up, min=-limit, max=limit)
    return F.silu(g) * u


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def sinkhorn_mhc_mix(
    x_streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_streams: [B, 4, D], hc_fn: [24, 4*D], hc_base: [24], hc_scale: [3]
    bsz, c, d = x_streams.shape
    x_flat = x_streams.reshape(bsz, c * d)
    r_var = x_flat.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_flat * torch.rsqrt(r_var + eps)
    raw = x_norm @ hc_fn.t()  # [B, 24]
    h_pre = torch.sigmoid(raw[:, :c] * hc_scale[0] + hc_base[:c])  # [B, 4]
    h_post = 2.0 * torch.sigmoid(raw[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c])  # [B, 4]
    h_res_logits = (raw[:, 2 * c :] * hc_scale[2] + hc_base[2 * c :]).reshape(bsz, c, c)
    m = torch.exp(h_res_logits - h_res_logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    x_in = torch.einsum("bc,bcd->bd", h_pre, x_streams)
    return x_in, h_post, m


def sinkhorn_mhc_merge(
    x_streams: torch.Tensor, y_branch: torch.Tensor, h_post: torch.Tensor, m: torch.Tensor
) -> torch.Tensor:
    # m: [B, C, C], x_streams: [B, C, D], h_post: [B, C], y_branch: [B, D]
    mixed = torch.einsum("bij,bjd->bid", m, x_streams)
    return mixed + h_post.unsqueeze(-1) * y_branch.unsqueeze(1)


def compute_metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    diff = ref - cand
    rel_l2 = float(torch.linalg.norm(diff) / (torch.linalg.norm(ref) + 1e-12))
    cos = float(F.cosine_similarity(ref.reshape(-1), cand.reshape(-1), dim=0))
    return {"rel_l2": rel_l2, "cosine": cos}


def run_eval(snap_dir: Path, num_tokens: int = 32) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[eval_hq_kld] Scanning safetensors in {snap_dir} on device={device}...", flush=True)
    locator = build_tensor_locator(snap_dir)
    print(f"[eval_hq_kld] Indexed {len(locator)} tensors across downloaded shards.", flush=True)

    torch.manual_seed(2026)
    embed_w = load_tensor_f32(locator, "model.language_model.embed_tokens.weight", device)
    vocab_size, hidden = embed_w.shape
    sample_ids = torch.randint(100, vocab_size - 100, (num_tokens,), device=device)
    x0 = embed_w[sample_ids] * math.sqrt(hidden) * 0.02 + embed_w[sample_ids]

    results: dict = {"stage1": {}, "stage2": {}}

    # ── Stage 1A: Single Matrix Clip-Shrink Ablation on Layer 3 Expert 0 ──
    e0_down = load_tensor_f32(locator, "model.language_model.layers.3.mlp.experts.0.down_proj.weight", device)
    e0_gate = load_tensor_f32(locator, "model.language_model.layers.3.mlp.experts.0.gate_proj.weight", device)
    ablation = {}
    for spec in ("hq2_g128", "hq2_g64", "hq2_g32", "hq3_g256", "hq3_g128", "hq3_g64", "hq4_g32", "hq6_g32"):
        b, g, _ = parse_quant_spec(spec)
        bpw = hq_block_bytes(b, g) * 8.0 / g
        dq_noclip = quant_dequant_torch(e0_down, spec, mse_clip=False)
        dq_clip = quant_dequant_torch(e0_down, spec, mse_clip=True)
        ablation[spec] = {
            "bpw": bpw,
            "no_clip": compute_metrics(e0_down, dq_noclip),
            "mse_clip": compute_metrics(e0_down, dq_clip),
        }
        print(
            f"  [Matrix Ablation] {spec:8s} ({bpw:.3f} bpw): "
            f"no_clip rel_l2={ablation[spec]['no_clip']['rel_l2']:.4f} -> "
            f"mse_clip rel_l2={ablation[spec]['mse_clip']['rel_l2']:.4f} "
            f"(cos={ablation[spec]['mse_clip']['cosine']:.5f})",
            flush=True,
        )
    results["stage1"]["matrix_ablation"] = ablation

    # ── Stage 1B: Dense FFN Layer 0 (Clamped SwiGLU limit=10.0) ──
    l0_ln2 = load_tensor_f32(locator, "model.language_model.layers.0.post_attention_layernorm.weight", device)
    l0_g = load_tensor_f32(locator, "model.language_model.layers.0.mlp.gate_proj.weight", device)
    l0_u = load_tensor_f32(locator, "model.language_model.layers.0.mlp.up_proj.weight", device)
    l0_d = load_tensor_f32(locator, "model.language_model.layers.0.mlp.down_proj.weight", device)
    h_l0 = rms_norm(x0, l0_ln2)
    y_l0_ref = clamped_swiglu(h_l0 @ l0_g.t(), h_l0 @ l0_u.t(), 10.0) @ l0_d.t()
    l0_res = {}
    for g_spec, d_spec in [("hq6_g32", "hq6_G32"), ("hq4_g32", "hq6_g32"), ("hq4_g32", "hq4_g32")]:
        g_spec_l, d_spec_l = g_spec.lower(), d_spec.lower()
        qg = quant_dequant_torch(l0_g, g_spec_l)
        qu = quant_dequant_torch(l0_u, g_spec_l)
        qd = quant_dequant_torch(l0_d, d_spec_l)
        y_q = clamped_swiglu(h_l0 @ qg.t(), h_l0 @ qu.t(), 10.0) @ qd.t()
        key = f"gu_{g_spec_l}__down_{d_spec_l}"
        l0_res[key] = compute_metrics(y_l0_ref, y_q)
        print(f"  [Layer 0 Dense FFN] {key}: {l0_res[key]}", flush=True)
    results["stage1"]["layer0_dense_ffn"] = l0_res

    # ── Stage 1C: Layer 3 Router Bias & 288-Expert MoE Activation Analysis ──
    l3_ln2 = load_tensor_f32(locator, "model.language_model.layers.3.post_attention_layernorm.weight", device)
    l3_gate_w = load_tensor_f32(locator, "model.language_model.layers.3.mlp.gate.weight", device)
    l3_bias = load_tensor_f32(locator, "model.language_model.layers.3.mlp.gate.e_score_correction_bias", device)
    h_l3 = rms_norm(x0 + y_l0_ref, l3_ln2)

    # Generate a larger synthetic token probe (512 tokens) to measure router hit concentration
    probe_ids = torch.randint(100, vocab_size - 100, (512,), device=device)
    h_probe = rms_norm(embed_w[probe_ids], l3_ln2)
    scores_probe = torch.sigmoid(h_probe @ l3_gate_w.t())
    _, topk_probe = torch.topk(scores_probe + l3_bias.unsqueeze(0), k=8, dim=-1)
    counts = torch.bincount(topk_probe.reshape(-1), minlength=288).float()
    sorted_counts, _ = torch.sort(counts, descending=True)
    total_hits = float(counts.sum())
    hit_concentration = {
        "top_48_share": float(sorted_counts[:48].sum() / total_hits),
        "top_96_share": float(sorted_counts[:96].sum() / total_hits),
        "top_144_share": float(sorted_counts[:144].sum() / total_hits),
        "top_216_share": float(sorted_counts[:216].sum() / total_hits),
    }
    print(f"  [Layer 3 Router Concentration (288 experts)]: {hit_concentration}", flush=True)
    results["stage1"]["layer3_router_concentration"] = hit_concentration

    # Evaluate Layer 3 MoE output fidelity on active experts for our token batch
    scores = torch.sigmoid(h_l3 @ l3_gate_w.t())
    _, topk_idx = torch.topk(scores + l3_bias.unsqueeze(0), k=8, dim=-1)
    topk_unbiased = torch.gather(scores, 1, topk_idx)
    topk_weights = (topk_unbiased / (topk_unbiased.sum(dim=-1, keepdim=True) + 1e-20)) * 2.5

    sh_g = load_tensor_f32(locator, "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight", device)
    sh_u = load_tensor_f32(locator, "model.language_model.layers.3.mlp.shared_experts.up_proj.weight", device)
    sh_d = load_tensor_f32(locator, "model.language_model.layers.3.mlp.shared_experts.down_proj.weight", device)
    y_sh_ref = clamped_swiglu(h_l3 @ sh_g.t(), h_l3 @ sh_u.t(), 10.0) @ sh_d.t()

    unique_experts = sorted(set(topk_idx.reshape(-1).tolist()))
    num_u = len(unique_experts)
    print(f"  [Layer 3 MoE] Loading {num_u} unique active experts for {num_tokens} tokens...", flush=True)
    g_list, u_list, d_list = [], [], []
    for e in unique_experts:
        g_list.append(load_tensor_f32(locator, f"model.language_model.layers.3.mlp.experts.{e}.gate_proj.weight", device))
        u_list.append(load_tensor_f32(locator, f"model.language_model.layers.3.mlp.experts.{e}.up_proj.weight", device))
        d_list.append(load_tensor_f32(locator, f"model.language_model.layers.3.mlp.experts.{e}.down_proj.weight", device))
    g_stack = torch.stack(g_list, dim=0)  # [U, 2048, 4096]
    u_stack = torch.stack(u_list, dim=0)  # [U, 2048, 4096]
    d_stack = torch.stack(d_list, dim=0)  # [U, 4096, 2048]

    def batch_qd(stack: torch.Tensor, spec: str) -> torch.Tensor:
        u_cnt, r_cnt, c_cnt = stack.shape
        return quant_dequant_torch(stack.reshape(u_cnt * r_cnt, c_cnt), spec).reshape(u_cnt, r_cnt, c_cnt)

    g_q = {s: batch_qd(g_stack, s) for s in ("hq3_g128", "hq2_g64", "hq2_g128")}
    u_q = {s: batch_qd(u_stack, s) for s in ("hq3_g128", "hq2_g64", "hq2_g128")}
    d_q = {s: batch_qd(d_stack, s) for s in ("hq3_g128", "hq2_g32", "hq2_g64", "hq2_g128")}

    # Rank experts by router prior (e_score_correction_bias + empirical hits)
    prior_order = torch.argsort(l3_bias + counts * 0.01, descending=True).tolist()
    expert_rank = {e: r for r, e in enumerate(prior_order)}

    sh_gq = quant_dequant_torch(sh_g, "hq6_g32")
    sh_uq = quant_dequant_torch(sh_u, "hq6_g32")
    sh_dq = quant_dequant_torch(sh_d, "hq6_g32")
    y_sh_q = clamped_swiglu(h_l3 @ sh_gq.t(), h_l3 @ sh_uq.t(), 10.0) @ sh_dq.t()

    def eval_moe_policy(quant_fn) -> torch.Tensor:
        y_out = y_sh_q.clone()
        for idx_u, e in enumerate(unique_experts):
            mask = topk_idx == e
            if not mask.any():
                continue
            tok_indices, slot_indices = torch.where(mask)
            x_e = h_l3[tok_indices]
            w_e = topk_weights[tok_indices, slot_indices].unsqueeze(-1)
            qg, qu, qd = quant_fn(idx_u, e)
            y_e = clamped_swiglu(x_e @ qg.t(), x_e @ qu.t(), 10.0) @ qd.t()
            y_out.index_add_(0, tok_indices, w_e * y_e)
        return y_out

    # Reference teacher MoE output
    y_moe_ref = y_sh_ref.clone()
    for idx_u, e in enumerate(unique_experts):
        mask = topk_idx == e
        tok_indices, slot_indices = torch.where(mask)
        x_e = h_l3[tok_indices]
        w_e = topk_weights[tok_indices, slot_indices].unsqueeze(-1)
        eg, eu, ed = g_stack[idx_u], u_stack[idx_u], d_stack[idx_u]
        y_e = clamped_swiglu(x_e @ eg.t(), x_e @ eu.t(), 10.0) @ ed.t()
        y_moe_ref.index_add_(0, tok_indices, w_e * y_e)

    policies = {
        "uniform_hq3_g128 (115.0 GiB exp)": lambda idx_u, e: (
            g_q["hq3_g128"][idx_u],
            u_q["hq3_g128"][idx_u],
            d_q["hq3_g128"][idx_u],
        ),
        "gu_hq2_g128__down_hq3_g128 (91.4 GiB exp)": lambda idx_u, e: (
            g_q["hq2_g128"][idx_u],
            u_q["hq2_g128"][idx_u],
            d_q["hq3_g128"][idx_u],
        ),
        "gu_hq2_g128__down_hq2_g32 (88.5 GiB exp)": lambda idx_u, e: (
            g_q["hq2_g128"][idx_u],
            u_q["hq2_g128"][idx_u],
            d_q["hq2_g32"][idx_u],
        ),
        "tiered_hot_warm_cold (83.2 GiB exp)": lambda idx_u, e: (
            g_q["hq2_g64" if expert_rank[e] < 48 else "hq2_g128"][idx_u],
            u_q["hq2_g64" if expert_rank[e] < 48 else "hq2_g128"][idx_u],
            d_q["hq3_g128" if expert_rank[e] < 64 else ("hq2_g64" if expert_rank[e] < 144 else "hq2_g128")][idx_u],
        ),
        "uniform_hq2_g128 (79.6 GiB exp)": lambda idx_u, e: (
            g_q["hq2_g128"][idx_u],
            u_q["hq2_g128"][idx_u],
            d_q["hq2_g128"][idx_u],
        ),
    }

    # ── Stage 2: End-to-End Logit KLD & Top-1 Agreement Gate ──
    norm_key = (
        "model.language_model.norm.weight"
        if "model.language_model.norm.weight" in locator
        else "model.language_model.layers.45.shared_head.norm.weight"
    )
    final_norm_w = load_tensor_f32(locator, norm_key, device)
    lm_head_w = load_tensor_f32(locator, "lm_head.weight", device)
    lm_head_q6 = quant_dequant_torch(lm_head_w, "hq6_g32")

    # Teacher hidden state after residual + final norm -> teacher logits
    h_final_ref = rms_norm(x0 + y_l0_ref + y_moe_ref, final_norm_w)
    logits_ref = h_final_ref @ lm_head_w.t()
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
    probs_ref = log_probs_ref.exp()
    top1_ref = logits_ref.argmax(dim=-1)

    l0_gq = quant_dequant_torch(l0_g, "hq4_g32")
    l0_uq = quant_dequant_torch(l0_u, "hq4_g32")
    l0_dq = quant_dequant_torch(l0_d, "hq6_g32")
    y_l0_q = clamped_swiglu(h_l0 @ l0_gq.t(), h_l0 @ l0_uq.t(), 10.0) @ l0_dq.t()

    moe_results = {}
    stage2_results = {}
    for pname, pfn in policies.items():
        y_moe_q = eval_moe_policy(pfn)
        m_moe = compute_metrics(y_moe_ref, y_moe_q)
        moe_results[pname] = m_moe

        h_final_q = rms_norm(x0 + y_l0_q + y_moe_q, final_norm_w)
        logits_q = h_final_q @ lm_head_q6.t()
        log_probs_q = F.log_softmax(logits_q, dim=-1)
        kld_per_tok = (probs_ref * (log_probs_ref - log_probs_q)).sum(dim=-1)
        mean_kld = float(kld_per_tok.mean())
        p99_kld = float(torch.quantile(kld_per_tok, 0.99))
        top1_acc = float((logits_q.argmax(dim=-1) == top1_ref).float().mean())
        stage2_results[pname] = {
            "moe_rel_l2": m_moe["rel_l2"],
            "moe_cosine": m_moe["cosine"],
            "mean_kld": mean_kld,
            "p99_kld": p99_kld,
            "top1_agreement": top1_acc,
        }
        print(
            f"  [Stage 2 KLD] {pname:42s}: "
            f"MoE cos={m_moe['cosine']:.5f} | mean_KLD={mean_kld:.6f} | "
            f"p99_KLD={p99_kld:.6f} | top1={top1_acc*100:.1f}%",
            flush=True,
        )

    results["stage1"]["layer3_moe_policies"] = moe_results
    results["stage2"]["logit_kld_gate"] = stage2_results
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-cache",
        default="~/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash",
        help="Path to HuggingFace cache directory for zai-org/GLM-5.3-Flash",
    )
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    snaps = sorted(glob.glob(os.path.expanduser(args.model_cache) + "/snapshots/*"))
    if not snaps:
        sys.exit(f"No snapshots found under {args.model_cache}")
    res = run_eval(Path(snaps[-1]), num_tokens=args.tokens)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[eval_hq_kld] Wrote results to {args.out}", flush=True)
