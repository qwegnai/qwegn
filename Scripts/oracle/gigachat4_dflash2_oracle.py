#!/usr/bin/env python3
"""2-layer synthetic PyTorch SGLang reference oracle for GigaChat4DFlash2DraftModel."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def f16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).float()


def gated_norm(
    x: torch.Tensor, w: torch.Tensor, gu: torch.Tensor, gd: torch.Tensor,
    eps: float = 1e-6, gw: float = 2.0, scale: float = 1.0, out_f16: bool = True,
) -> torch.Tensor:
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    h = bf16_round(xf * rstd * w.float())
    g1 = bf16_round(F.silu(F.linear(h, gu.float())))
    g2 = bf16_round(torch.sigmoid(F.linear(g1, gd.float())))
    y = bf16_round(h * gw * g2)
    if scale != 1.0:
        y = bf16_round(y * scale)
    return f16_round(y) if out_f16 else y


def grouped_conv(x: torch.Tensor, delta: torch.Tensor, base_side: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    m_rows, h_dim = x.shape
    groups = h_dim // group_size
    blocks = x.float().view(m_rows, groups, group_size)
    coeff = base_side.half().float().view(1, 2, groups, group_size) + delta.float().unsqueeze(-1)
    shifted = torch.zeros_like(blocks)
    shifted[1:] = blocks[:-1]
    return f16_round((coeff[:, 0] * blocks + coeff[:, 1] * shifted).reshape(m_rows, h_dim))


def compute_yarn_inv_freq(
    dim: int, base: float, factor: float, orig_max: int,
    beta_fast: float = 32.0, beta_slow: float = 1.0, mscale: float = 1.0, mscale_all_dim: float = 1.0,
):
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    low = max(math.floor(dim * math.log(orig_max / (beta_fast * 2 * math.pi)) / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(orig_max / (beta_slow * 2 * math.pi)) / (2 * math.log(base))), dim // 2 - 1)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float64) - low) / max(high - low, 0.001), 0.0, 1.0)
    inv_freq = inv_interp * ramp + inv_extrap * (1.0 - ramp)
    ms = ((0.1 * mscale * math.log(factor) + 1.0) / (0.1 * mscale_all_dim * math.log(factor) + 1.0)) if factor > 1.0 else 1.0
    return inv_freq.float(), float(ms)


def apply_rope_interleave(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, mscale: float) -> torch.Tensor:
    ang = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    shape = [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1] // 2]
    c = (torch.cos(ang) * mscale).view(*shape)
    s = (torch.sin(ang) * mscale).view(*shape)
    x0, x1 = x.float()[..., 0::2], x.float()[..., 1::2]
    y = torch.empty_like(x.float())
    y[..., 0::2] = x0 * c - x1 * s
    y[..., 1::2] = x1 * c + x0 * s
    return f16_round(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)

    H, I, V, L, q_heads = 128, 256, 64, 2, 2
    q_lora, kv_lora, nope_dim, rope_dim, v_dim = 64, 128, 64, 64, 64
    sel_rank, n_rank, conv_group_size = 32, 16, 16
    conv_groups = H // conv_group_size
    conv_proj_dim = 4 * conv_groups
    swiglu_limit = 2.0
    alpha_q, alpha_kv = math.sqrt(H / q_lora), math.sqrt(H / kv_lora)
    inv_freq, rope_mscale = compute_yarn_inv_freq(rope_dim, 100000.0, 32.0, 32768, 32.0, 1.0, 1.0, 1.0)
    m_all = 0.1 * 1.0 * math.log(32.0) + 1.0
    softmax_scale = (1.0 / math.sqrt(nope_dim + rope_dim)) * m_all * m_all

    cfg = {
        "architectures": ["GigaChat4DFlash2DraftModel"],
        "model_type": "gigachat4",
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": L,
        "num_attention_heads": q_heads,
        "num_key_value_heads": q_heads,
        "qk_head_dim": nope_dim + rope_dim,
        "qk_nope_head_dim": nope_dim,
        "qk_rope_head_dim": rope_dim,
        "v_head_dim": v_dim,
        "q_lora_rank": q_lora,
        "kv_lora_rank": kv_lora,
        "gated_attention": True,
        "use_mla_scaling_factor": True,
        "norm_type": "GatedNorm",
        "layernorm_gating_weight": 2.0,
        "layernorm_type": "pre_post",
        "swiglu_limit": swiglu_limit,
        "vocab_size": V,
        "sliding_window": 32,
        "final_logit_softcapping": 30.0,
        "rope_interleave": True,
        "rope_parameters": {
            "rope_type": "yarn", "rope_theta": 100000.0, "factor": 32.0,
            "original_max_position_embeddings": 32768, "beta_fast": 32, "beta_slow": 1,
            "mscale": 1.0, "mscale_all_dim": 1.0,
        },
        "dflash_config": {
            "attention_mode": "mla", "block_size": 8, "mask_token_id": 63, "shift_label": True,
            "target_layer_ids": [0, 2], "conv_kernel_size": 2, "conv_group_size": conv_group_size,
            "selector_rank": sel_rank, "selector_top_k": 16, "final_logit_softcapping": 30.0,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def rand_bf16(shape, std=0.15):
        return (torch.randn(shape, dtype=torch.float32) * std).to(torch.bfloat16)

    st = {}
    def add_gn(pfx: str, dim: int):
        st[f"{pfx}.weight"] = (0.9 + 0.1 * torch.randn(dim)).to(torch.bfloat16)
        st[f"{pfx}.gate_up_projection.weight"] = rand_bf16((n_rank, dim), 0.2)
        st[f"{pfx}.gate_down_projection.weight"] = rand_bf16((dim, n_rank), 0.2)

    st["fc.weight"] = rand_bf16((H, 2 * H), 0.15)
    add_gn("hidden_norm", H)
    add_gn("norm", H)
    st["candidate_selector.hidden_projection.weight"] = rand_bf16((sel_rank, H), 0.2)
    st["candidate_selector.predecessor_codebook"] = rand_bf16((V, sel_rank), 0.3)
    st["candidate_selector.successor_codebook"] = rand_bf16((V, sel_rank), 0.3)

    for l in range(L):
        p = f"layers.{l}."
        add_gn(p + "input_layernorm", H)
        add_gn(p + "self_attn.q_a_layernorm", q_lora)
        add_gn(p + "self_attn.kv_a_layernorm", kv_lora)
        add_gn(p + "post_self_attn_layernorm", H)
        add_gn(p + "post_attention_layernorm", H)
        add_gn(p + "post_feedforward_layernorm", H)
        for cname in ("attention_conv", "mlp_conv"):
            bk = torch.zeros(2, 2, H, dtype=torch.float32)
            bk[:, 0] = 1.0 + 0.05 * torch.randn(2, H)
            bk[:, 1] = 0.25 * torch.randn(2, H)
            st[p + f"{cname}.base_kernel"] = bk.to(torch.bfloat16)
            st[p + f"{cname}.kernel_projection.weight"] = rand_bf16((conv_proj_dim, H), 0.02)
        st[p + "self_attn.q_a_proj.weight"] = rand_bf16((q_lora, H), 0.15)
        st[p + "self_attn.kv_a_proj_with_mqa.weight"] = rand_bf16((kv_lora + rope_dim, H), 0.15)
        st[p + "self_attn.gate_proj.weight"] = rand_bf16((q_heads * v_dim, H), 0.2)
        st[p + "self_attn.q_b_proj.weight"] = rand_bf16((q_heads * (nope_dim + rope_dim), q_lora), 0.15)
        st[p + "self_attn.kv_b_proj.weight"] = rand_bf16((q_heads * (nope_dim + v_dim), kv_lora), 0.15)
        st[p + "self_attn.o_proj.weight"] = rand_bf16((H, q_heads * v_dim), 0.15)
        st[p + "mlp.gate_proj.weight"] = rand_bf16((I, H), 0.15)
        st[p + "mlp.up_proj.weight"] = rand_bf16((I, H), 0.15)
        st[p + "mlp.down_proj.weight"] = rand_bf16((H, I), 0.15)
    save_file(st, str(out_dir / "model.safetensors"))

    embed_w = rand_bf16((V, H), 0.25)
    head_w = rand_bf16((V, H), 0.25).to(torch.float16)
    ctx_rows, M, D, anchor_id, mask_id = 4, 8, 7, 5, 63
    tap0 = (torch.randn(ctx_rows, H) * 0.5).to(torch.float16)
    tap1 = (torch.randn(ctx_rows, H) * 0.5).to(torch.float16)
    h_ctx = gated_norm(
        f16_round(F.linear(torch.cat([tap0.float(), tap1.float()], dim=-1), st["fc.weight"].float())),
        st["hidden_norm.weight"], st["hidden_norm.gate_up_projection.weight"], st["hidden_norm.gate_down_projection.weight"],
    )

    prop_x = embed_w[[anchor_id] + [mask_id] * D].float()
    h = gated_norm(
        prop_x, st["layers.0.input_layernorm.weight"],
        st["layers.0.input_layernorm.gate_up_projection.weight"],
        st["layers.0.input_layernorm.gate_down_projection.weight"],
    )
    ctx_pos, prop_pos = torch.arange(ctx_rows), torch.arange(ctx_rows, ctx_rows + M)

    for l in range(L):
        p = f"layers.{l}."
        kv_ctx = f16_round(F.linear(h_ctx, st[p + "self_attn.kv_a_proj_with_mqa.weight"].float()))
        c_kv_ctx = gated_norm(
            kv_ctx[:, :kv_lora], st[p + "self_attn.kv_a_layernorm.weight"],
            st[p + "self_attn.kv_a_layernorm.gate_up_projection.weight"],
            st[p + "self_attn.kv_a_layernorm.gate_down_projection.weight"], scale=alpha_kv,
        )
        k_pe_ctx = apply_rope_interleave(kv_ctx[:, kv_lora:], ctx_pos, inv_freq, rope_mscale)
        ac = F.linear(h, st[p + "attention_conv.kernel_projection.weight"].half().float()).view(M, 2, 2, conv_groups)
        h_a = grouped_conv(h, ac[:, 0], st[p + "attention_conv.base_kernel"][0], conv_group_size)
        qkv = f16_round(F.linear(h_a, torch.cat([
            st[p + "self_attn.q_a_proj.weight"], st[p + "self_attn.kv_a_proj_with_mqa.weight"], st[p + "self_attn.gate_proj.weight"]
        ], dim=0).float()))
        q_lat = gated_norm(
            qkv[:, :q_lora], st[p + "self_attn.q_a_layernorm.weight"],
            st[p + "self_attn.q_a_layernorm.gate_up_projection.weight"],
            st[p + "self_attn.q_a_layernorm.gate_down_projection.weight"], scale=alpha_q,
        )
        q_full = f16_round(F.linear(q_lat, st[p + "self_attn.q_b_proj.weight"].float())).view(M, q_heads, nope_dim + rope_dim)
        q_nope = q_full[:, :, :nope_dim]
        q_rope = apply_rope_interleave(q_full[:, :, nope_dim:], prop_pos, inv_freq, rope_mscale)
        c_kv_p = gated_norm(
            qkv[:, q_lora:q_lora + kv_lora], st[p + "self_attn.kv_a_layernorm.weight"],
            st[p + "self_attn.kv_a_layernorm.gate_up_projection.weight"],
            st[p + "self_attn.kv_a_layernorm.gate_down_projection.weight"], scale=alpha_kv,
        )
        k_pe_p = apply_rope_interleave(qkv[:, q_lora + kv_lora:q_lora + kv_lora + rope_dim], prop_pos, inv_freq, rope_mscale)
        kv_b = st[p + "self_attn.kv_b_proj.weight"].float().view(q_heads, nope_dim + v_dim, kv_lora)
        q_abs = f16_round(torch.einsum("mhd,hdk->mhk", q_nope, kv_b[:, :nope_dim]))
        scores = (
            torch.einsum("mhk,tk->mht", q_abs, torch.cat([c_kv_ctx, c_kv_p], dim=0))
            + torch.einsum("mhr,tr->mht", q_rope, torch.cat([k_pe_ctx, k_pe_p], dim=0))
        ) * softmax_scale
        u_lat = f16_round(torch.einsum("mht,tk->mhk", torch.softmax(scores, dim=-1), torch.cat([c_kv_ctx, c_kv_p], dim=0)))
        attn_val = torch.einsum("mhk,hvk->mhv", u_lat, kv_b[:, nope_dim:]).reshape(M, q_heads * v_dim)
        attn_gated = f16_round(attn_val * torch.sigmoid(qkv[:, q_lora + kv_lora + rope_dim:]))
        p_down = f16_round(F.linear(attn_gated, st[p + "self_attn.o_proj.weight"].float()))
        p_norm = grouped_conv(p_down, ac[:, 1], st[p + "attention_conv.base_kernel"][1], conv_group_size)
        prop_x = prop_x + gated_norm(
            p_norm, st[p + "post_self_attn_layernorm.weight"],
            st[p + "post_self_attn_layernorm.gate_up_projection.weight"],
            st[p + "post_self_attn_layernorm.gate_down_projection.weight"], out_f16=False,
        )
        h = gated_norm(
            prop_x, st[p + "post_attention_layernorm.weight"],
            st[p + "post_attention_layernorm.gate_up_projection.weight"],
            st[p + "post_attention_layernorm.gate_down_projection.weight"],
        )
        mc = F.linear(h, st[p + "mlp_conv.kernel_projection.weight"].half().float()).view(M, 2, 2, conv_groups)
        h_m = grouped_conv(h, mc[:, 0], st[p + "mlp_conv.base_kernel"][0], conv_group_size)
        gu = f16_round(F.linear(h_m, torch.cat([st[p + "mlp.gate_proj.weight"], st[p + "mlp.up_proj.weight"]], dim=0).float()))
        mlp_inter = f16_round(F.silu(gu[:, :I].clamp(max=swiglu_limit)) * gu[:, I:].clamp(-swiglu_limit, swiglu_limit))
        mlp_down = f16_round(F.linear(mlp_inter, st[p + "mlp.down_proj.weight"].float()))
        mlp_fin = grouped_conv(mlp_down, mc[:, 1], st[p + "mlp_conv.base_kernel"][1], conv_group_size)
        prop_x = prop_x + gated_norm(
            mlp_fin, st[p + "post_feedforward_layernorm.weight"],
            st[p + "post_feedforward_layernorm.gate_up_projection.weight"],
            st[p + "post_feedforward_layernorm.gate_down_projection.weight"], out_f16=False,
        )
        if l + 1 < L:
            h = gated_norm(
                prop_x, st[f"layers.{l + 1}.input_layernorm.weight"],
                st[f"layers.{l + 1}.input_layernorm.gate_up_projection.weight"],
                st[f"layers.{l + 1}.input_layernorm.gate_down_projection.weight"],
            )

    pred_hidden = gated_norm(
        prop_x[:-1], st["norm.weight"],
        st["norm.gate_up_projection.weight"], st["norm.gate_down_projection.weight"],
    )
    raw_logits = F.linear(pred_hidden, head_w.float())
    top_vals, top_ids = torch.topk(raw_logits, 16, dim=-1)
    unary = torch.tanh(top_vals / 30.0) * 30.0
    h_sel = F.linear(pred_hidden, st["candidate_selector.hidden_projection.weight"].half().float())
    pred_book = st["candidate_selector.predecessor_codebook"].float()
    succ_book = st["candidate_selector.successor_codebook"].float()

    chosen, cur_pred = [], anchor_id
    for pos in range(D):
        cands = top_ids[pos]
        edge = ((pred_book[cur_pred] * h_sel[pos]).unsqueeze(0) * succ_book[cands]).sum(dim=-1)
        win = int(cands[int((unary[pos] + edge).argmax())])
        chosen.append(win)
        cur_pred = win

    (out_dir / "embed_bf16.bin").write_bytes(embed_w.view(torch.uint16).numpy().tobytes())
    (out_dir / "head_f16.bin").write_bytes(head_w.view(torch.uint16).numpy().tobytes())
    (out_dir / "tap0_f16.bin").write_bytes(tap0.view(torch.uint16).numpy().tobytes())
    (out_dir / "tap1_f16.bin").write_bytes(tap1.view(torch.uint16).numpy().tobytes())
    (out_dir / "pred_hidden.bin").write_bytes(pred_hidden.float().numpy().astype(np.float32).tobytes())
    (out_dir / "draft_tokens.bin").write_bytes(np.array(chosen, dtype=np.int32).tobytes())


if __name__ == "__main__":
    main()
