"""GigaChat4 QWP1 packer adapter (hybrid MLA-DSA + GDN + GatedNorm + Group-MoE + NextN MTP)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ernie55_adapter import ShardReader, _hq_meta
from hq_blocks import validate_source_quant_config
from pack import pack_file, resolve_snapshot_dir, sha256, synthesize_base_chat_template
from shvq import any_block_bytes, format_any_quant_spec, parse_any_quant_spec, quant_any_blocks
from verify import audit_pack, read_header

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402


def _deinterleave_mla_q(q_raw, q_heads: int, nope_d: int, rope_d: int, in_dim: int):
    viewed = q_raw.view(q_heads, nope_d + rope_d, in_dim)
    q_nope = viewed[:, :nope_d, :].reshape(q_heads * nope_d, in_dim).contiguous()
    q_rope = viewed[:, nope_d:, :].reshape(q_heads * rope_d, in_dim).contiguous()
    return q_nope, q_rope


def _deinterleave_gdn_qkvz_ba(
    qkvz,
    ba,
    *,
    qk_heads: int,
    v_heads: int,
    k_dim: int,
    v_dim: int,
    hidden: int,
):
    """Permute Megatron per-k-head [Q,K,V,Z] and [b,a] into planar [Q;K;V], [Z], [A], [B]."""
    import torch

    v_per_k = v_heads // qk_heads
    grp_qkvz = 2 * k_dim + 2 * v_per_k * v_dim
    gv = qkvz.view(qk_heads, grp_qkvz, hidden)
    q, k, v, z = torch.split(gv, [k_dim, k_dim, v_per_k * v_dim, v_per_k * v_dim], dim=1)
    in_proj_qkv = torch.cat([
        q.reshape(qk_heads * k_dim, hidden),
        k.reshape(qk_heads * k_dim, hidden),
        v.reshape(v_heads * v_dim, hidden),
    ], dim=0).contiguous()
    in_proj_z = z.reshape(v_heads * v_dim, hidden).contiguous()

    # modeling_gigachat4.py splits in_proj_ba into (b, a) in that order; Qwegn GDN uses a then b.
    bav = ba.view(qk_heads, 2 * v_per_k, hidden)
    b, a = torch.split(bav, [v_per_k, v_per_k], dim=1)
    in_proj_a = a.reshape(v_heads, hidden).contiguous()
    in_proj_b = b.reshape(v_heads, hidden).contiguous()
    return in_proj_qkv, in_proj_z, in_proj_a, in_proj_b


def _add_norm_shapes(exp: dict[str, list[int]], reader: ShardReader, prefix: str, dim: int) -> None:
    exp[prefix + "weight"] = [dim]
    if reader.has(prefix + "gate_up_projection.weight"):
        rank = int(reader.shapes[prefix + "gate_up_projection.weight"][0])
        exp[prefix + "gate_up_projection.weight"] = [rank, dim]
        exp[prefix + "gate_down_projection.weight"] = [dim, rank]


def _yield_norm(reader: ShardReader, prefix: str, bump):
    if reader.has(prefix + "gate_up_projection.weight"):
        bump("bf16", 3)
        yield prefix + "weight", reader.get(prefix + "weight"), "bf16"
        yield prefix + "gate_up_projection.weight", reader.get(prefix + "gate_up_projection.weight"), "bf16"
        yield prefix + "gate_down_projection.weight", reader.get(prefix + "gate_down_projection.weight"), "bf16"
    else:
        bump("bf16")
        yield prefix + "weight", reader.get(prefix + "weight"), "bf16"


def _is_mla_layer(tc: dict, layer: int) -> bool:
    ltypes = tc.get("layer_types")
    if isinstance(ltypes, list) and layer < len(ltypes):
        return ltypes[layer] in ("full_attention", "deepseek_sparse_attention")
    fa_layers = {int(x) for x in (tc.get("full_attention_layers") or [])}
    if tc.get("linear_attention_type") is not None or fa_layers:
        return layer in fa_layers
    return True


def build_gigachat4_expected_shapes(
    tc: dict,
    reader: ShardReader,
    *,
    dedup_mtp_tie: bool,
    tie_emb: bool = False,
) -> dict[str, list[int]]:
    h = int(tc["hidden_size"])
    vocab = int(tc["vocab_size"])
    num_layers = int(tc["num_hidden_layers"])
    inter = int(tc["intermediate_size"])
    moe_inter = int(tc.get("moe_intermediate_size") or inter)
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    n_sh = int(tc.get("n_shared_experts") or 0)
    sh_inter = n_sh * moe_inter
    first_k = int(tc.get("first_k_dense_replace") or 0)
    dense_attn = {int(x) for x in (tc.get("dense_attention_layers") or [])}
    use_sandwich = bool(tc.get("use_sandwich_norm") or tc.get("layernorm_type") == "pre_post")

    q_heads = int(tc["num_attention_heads"])
    q_rank = int(tc["q_lora_rank"])
    kv_rank = int(tc["kv_lora_rank"])
    nope_d = int(tc["qk_nope_head_dim"])
    rope_d = int(tc["qk_rope_head_dim"])
    v_d = int(tc["v_head_dim"])
    idx_h = int(tc.get("index_n_heads") or 0)
    idx_d = int(tc.get("index_head_dim") or 0)

    qk_h = int(tc.get("linear_num_key_heads") or 0)
    v_h = int(tc.get("linear_num_value_heads") or 0)
    gdn_hd = int(tc.get("linear_key_head_dim") or 0)
    conv_k = int(tc.get("linear_conv_kernel_dim") or 4)
    qkv_ch = (2 * qk_h + v_h) * gdn_hd
    z_ch = v_h * gdn_hd

    exp: dict[str, list[int]] = {"model.embed_tokens.weight": [vocab, h]}
    if not tie_emb or reader.has("lm_head.weight"):
        exp["lm_head.weight"] = [vocab, h]
    _add_norm_shapes(exp, reader, "model.norm.", h)

    def add_layer(l_idx: int, *, is_mla: bool, has_dsa: bool, is_moe: bool) -> None:
        blk = f"model.layers.{l_idx}."
        norm_names = (
            ("input_layernorm.", "post_self_attn_layernorm.", "post_attention_layernorm.", "post_feedforward_layernorm.")
            if use_sandwich
            else ("input_layernorm.", "post_attention_layernorm.")
        )
        for nm in norm_names:
            _add_norm_shapes(exp, reader, blk + nm, h)
        if is_mla:
            ap = blk + "self_attn."
            exp[ap + "q_a_proj.weight"] = [q_rank, h]
            _add_norm_shapes(exp, reader, ap + "q_a_layernorm.", q_rank)
            exp[ap + "q_b_nope_proj.weight"] = [q_heads * nope_d, q_rank]
            exp[ap + "q_b_rope_proj.weight"] = [q_heads * rope_d, q_rank]
            exp[ap + "kv_a_proj_with_mqa.weight"] = [kv_rank + rope_d, h]
            _add_norm_shapes(exp, reader, ap + "kv_a_layernorm.", kv_rank)
            exp[ap + "kv_b_proj.weight"] = [q_heads * (nope_d + v_d), kv_rank]
            if reader.has(ap + "gate_proj.weight"):
                exp[ap + "gate_proj.weight"] = [q_heads * v_d, h]
            exp[ap + "o_proj.weight"] = [h, q_heads * v_d]
            if has_dsa:
                exp[ap + "indexer.wk.weight"] = [idx_d, h]
                exp[ap + "indexer.wq_b.weight"] = [idx_h * idx_d, q_rank]
                exp[ap + "indexer.k_norm.weight"] = [idx_d]
                exp[ap + "indexer.weights_proj.weight"] = [idx_h, h]
        else:
            lp = blk + "linear_attn."
            exp[lp + "in_proj_qkv.weight"] = [qkv_ch, h]
            exp[lp + "in_proj_z.weight"] = [z_ch, h]
            exp[lp + "in_proj_a.weight"] = [v_h, h]
            exp[lp + "in_proj_b.weight"] = [v_h, h]
            exp[lp + "conv1d.weight"] = [qkv_ch, 1, conv_k]
            exp[lp + "norm.weight"] = [gdn_hd]
            exp[lp + "A_log"] = [v_h]
            exp[lp + "dt_bias"] = [v_h]
            exp[lp + "out_proj.weight"] = [h, z_ch]

        if is_moe and n_exp > 0:
            exp[blk + "mlp.gate.weight"] = [n_exp, h]
            exp[blk + "mlp.gate.e_score_correction_bias"] = [n_exp]
            if sh_inter > 0:
                exp[blk + "mlp.shared_experts.gate_proj.weight"] = [sh_inter, h]
                exp[blk + "mlp.shared_experts.up_proj.weight"] = [sh_inter, h]
                exp[blk + "mlp.shared_experts.down_proj.weight"] = [h, sh_inter]
            for e in range(n_exp):
                exp[blk + f"mlp.experts.{e}.gate_proj.weight"] = [moe_inter, h]
                exp[blk + f"mlp.experts.{e}.up_proj.weight"] = [moe_inter, h]
                exp[blk + f"mlp.experts.{e}.down_proj.weight"] = [h, moe_inter]
        else:
            exp[blk + "mlp.gate_proj.weight"] = [inter, h]
            exp[blk + "mlp.up_proj.weight"] = [inter, h]
            exp[blk + "mlp.down_proj.weight"] = [h, inter]

    for l in range(num_layers):
        is_mla = _is_mla_layer(tc, l)
        has_dsa = is_mla and idx_h > 0 and (l not in dense_attn)
        add_layer(l, is_mla=is_mla, has_dsa=has_dsa, is_moe=(l >= first_k and n_exp > 0))

    if int(tc.get("num_nextn_predict_layers") or 0) >= 1:
        mblk = f"model.layers.{num_layers}."
        _add_norm_shapes(exp, reader, mblk + "enorm.", h)
        _add_norm_shapes(exp, reader, mblk + "hnorm.", h)
        _add_norm_shapes(exp, reader, mblk + "shared_head.norm.", h)
        exp[mblk + "eh_proj.weight"] = [h, 2 * h]
        if not dedup_mtp_tie:
            exp[mblk + "embed_tokens.weight"] = [vocab, h]
            exp[mblk + "shared_head.head.weight"] = [vocab, h]
        nextn_is_mla = not bool(tc.get("nextn_is_linear", False))
        nextn_sparse = nextn_is_mla and bool(tc.get("nextn_is_sparse", False)) and idx_h > 0
        add_layer(num_layers, is_mla=nextn_is_mla, has_dsa=nextn_sparse, is_moe=False)

    return exp


def quantize_gigachat4_model(
    source: Path,
    destination: Path,
    flat_root: bool = False,
    quant: str = "hq4_g32_m5",
    group_size: int | None = None,
    dense_quant: str | None = None,
    expert_quant: str | None = None,
    expert_down_quant: str | None = None,
    reuse_source_sha_from: Path | None = None,
    expert_batch_size: int = 64,
    raw_gdn_ab: bool = False,
) -> dict[str, int]:
    import torch

    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    source = resolve_snapshot_dir(source).resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.suffix == ".qwp":
        raise ValueError("--out must be a model directory, not a .qwp file")

    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc = config["text_config"] if "text_config" in config else config
    parent_mt = config.get("model_type")
    mt = tc.get("model_type") or parent_mt or "gigachat4"
    fp8_block_size = validate_source_quant_config(config)

    if not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    embedded_tpl = None
    if not (source / "chat_template.jinja").is_file():
        tok_cfg_p = source / "tokenizer_config.json"
        if tok_cfg_p.is_file():
            tok_cfg = json.loads(tok_cfg_p.read_text())
            if isinstance(tok_cfg.get("chat_template"), str):
                embedded_tpl = tok_cfg["chat_template"]
        if not embedded_tpl:
            embedded_tpl = synthesize_base_chat_template(source)

    dense_spec = parse_any_quant_spec(dense_quant or quant, default_group=group_size)
    expert_spec = parse_any_quant_spec(expert_quant or quant, default_group=group_size)
    expert_down_spec = parse_any_quant_spec(expert_down_quant or expert_quant or quant, default_group=group_size)
    dense_fmt = format_any_quant_spec(*dense_spec)
    expert_fmt = format_any_quant_spec(*expert_spec)
    expert_down_fmt = format_any_quant_spec(*expert_down_spec)
    if dense_fmt == expert_fmt == expert_down_fmt:
        quant_recorded = dense_fmt
    elif expert_fmt == expert_down_fmt:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}"
    else:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}__expdown_{expert_down_fmt}"

    reader = ShardReader(source, files, fp8_block_size=fp8_block_size)
    num_layers = int(tc["num_hidden_layers"])
    tie_emb = bool(tc.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    mblk = f"model.layers.{num_layers}."
    has_mtp_emb = reader.has(mblk + "embed_tokens.weight")
    has_mtp_head = reader.has(mblk + "shared_head.head.weight")
    if has_mtp_emb and has_mtp_head:
        same_emb = bool(torch.equal(reader.get("model.embed_tokens.weight"), reader.get(mblk + "embed_tokens.weight")))
        same_head = (
            not reader.has("lm_head.weight")
            or bool(torch.equal(reader.get("lm_head.weight"), reader.get(mblk + "shared_head.head.weight")))
        )
        dedup_mtp_tie = same_emb and same_head
    else:
        dedup_mtp_tie = not has_mtp_emb and not has_mtp_head

    expected_shapes = build_gigachat4_expected_shapes(
        tc, reader, dedup_mtp_tie=dedup_mtp_tie, tie_emb=tie_emb
    )

    sha256_source_hex = None
    if reuse_source_sha_from is not None:
        rq = Path(reuse_source_sha_from) / "model.qwp"
        if rq.is_file():
            rp = read_header(rq)[0].get("provenance", {})
            if rp.get("source") == str(source) and rp.get("sha256_source"):
                sha256_source_hex = rp["sha256_source"]
    if sha256_source_hex is None:
        h_src = hashlib.sha256()
        for path in files:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(16 << 20), b""):
                    h_src.update(chunk)
        sha256_source_hex = h_src.hexdigest()

    geo_digest = hashlib.sha256(geo_v2(tc, parent_mt=parent_mt).encode()).hexdigest()
    stats: dict[str, int] = {}

    def bump(k: str, n: int = 1) -> None:
        stats[k] = stats.get(k, 0) + n

    h = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    moe_inter = int(tc.get("moe_intermediate_size") or inter)
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    n_sh = int(tc.get("n_shared_experts") or 0)
    sh_inter = n_sh * moe_inter
    first_k = int(tc.get("first_k_dense_replace") or 0)
    dense_attn = {int(x) for x in (tc.get("dense_attention_layers") or [])}
    use_sandwich = bool(tc.get("use_sandwich_norm") or tc.get("layernorm_type") == "pre_post")
    q_heads = int(tc["num_attention_heads"])
    q_rank = int(tc["q_lora_rank"])
    nope_d = int(tc["qk_nope_head_dim"])
    rope_d = int(tc["qk_rope_head_dim"])
    idx_h = int(tc.get("index_n_heads") or 0)
    qk_h = int(tc.get("linear_num_key_heads") or 0)
    v_h = int(tc.get("linear_num_value_heads") or 0)
    gdn_hd = int(tc.get("linear_key_head_dim") or 0)

    def emit_layer(l_idx: int, *, is_mla: bool, has_dsa: bool, is_moe: bool):
        blk = f"model.layers.{l_idx}."
        norm_names = (
            ("input_layernorm.", "post_self_attn_layernorm.", "post_attention_layernorm.", "post_feedforward_layernorm.")
            if use_sandwich
            else ("input_layernorm.", "post_attention_layernorm.")
        )
        for nm in norm_names:
            yield from _yield_norm(reader, blk + nm, bump)

        if is_mla:
            ap = blk + "self_attn."
            bump(dense_fmt)
            yield ap + "q_a_proj.weight", reader.get(ap + "q_a_proj.weight"), dense_spec
            yield from _yield_norm(reader, ap + "q_a_layernorm.", bump)

            if reader.has(ap + "q_b_proj.weight"):
                qb_nope, qb_rope = _deinterleave_mla_q(
                    reader.get(ap + "q_b_proj.weight"), q_heads, nope_d, rope_d, q_rank
                )
            else:
                qb_nope = reader.get(ap + "q_b_nope_proj.weight")
                qb_rope = reader.get(ap + "q_b_rope_proj.weight")
            bump(dense_fmt, 2)
            yield ap + "q_b_nope_proj.weight", qb_nope, dense_spec
            yield ap + "q_b_rope_proj.weight", qb_rope, dense_spec
            del qb_nope, qb_rope

            bump(dense_fmt)
            yield ap + "kv_a_proj_with_mqa.weight", reader.get(ap + "kv_a_proj_with_mqa.weight"), dense_spec
            yield from _yield_norm(reader, ap + "kv_a_layernorm.", bump)

            bump(dense_fmt, 2)
            yield ap + "kv_b_proj.weight", reader.get(ap + "kv_b_proj.weight"), dense_spec
            if reader.has(ap + "gate_proj.weight"):
                bump(dense_fmt)
                yield ap + "gate_proj.weight", reader.get(ap + "gate_proj.weight"), dense_spec
            yield ap + "o_proj.weight", reader.get(ap + "o_proj.weight"), dense_spec

            if has_dsa:
                bump(dense_fmt, 2)
                bump("bf16")
                bump("f32")
                yield ap + "indexer.wk.weight", reader.get(ap + "indexer.wk.weight"), dense_spec
                yield ap + "indexer.wq_b.weight", reader.get(ap + "indexer.wq_b.weight"), dense_spec
                yield ap + "indexer.k_norm.weight", reader.get(ap + "indexer.k_norm.weight"), "bf16"
                yield ap + "indexer.weights_proj.weight", reader.get(ap + "indexer.weights_proj.weight").to(torch.float32), "f32"
        else:
            sp = blk + ("self_attn." if reader.has(blk + "self_attn.in_proj_qkvz.weight") else "linear_attn.")
            lp = blk + "linear_attn."
            if reader.has(sp + "in_proj_qkvz.weight"):
                qkv, z, a, b = _deinterleave_gdn_qkvz_ba(
                    reader.get(sp + "in_proj_qkvz.weight"),
                    reader.get(sp + "in_proj_ba.weight"),
                    qk_heads=qk_h, v_heads=v_h, k_dim=gdn_hd, v_dim=gdn_hd, hidden=h,
                )
            else:
                qkv = reader.get(lp + "in_proj_qkv.weight")
                z = reader.get(lp + "in_proj_z.weight")
                a = reader.get(lp + "in_proj_a.weight")
                b = reader.get(lp + "in_proj_b.weight")
            ab_kind = "bf16" if raw_gdn_ab else dense_spec
            bump(dense_fmt, 3)
            bump("bf16" if raw_gdn_ab else dense_fmt, 2)
            bump("bf16", 2)
            bump("f32", 2)
            yield lp + "in_proj_qkv.weight", qkv, dense_spec
            yield lp + "in_proj_z.weight", z, dense_spec
            yield lp + "in_proj_a.weight", a, ab_kind
            yield lp + "in_proj_b.weight", b, ab_kind
            del qkv, z, a, b
            yield lp + "conv1d.weight", reader.get(sp + "conv1d.weight"), "bf16"
            yield lp + "norm.weight", reader.get(sp + "norm.weight"), "bf16"
            yield lp + "A_log", reader.get(sp + "A_log").to(torch.float32), "f32"
            yield lp + "dt_bias", reader.get(sp + "dt_bias").to(torch.float32), "f32"
            yield lp + "out_proj.weight", reader.get(sp + "out_proj.weight"), dense_spec

        if is_moe and n_exp > 0:
            bump("f32", 2)
            yield blk + "mlp.gate.weight", reader.get(blk + "mlp.gate.weight").to(torch.float32), "f32"
            bias_t = (
                reader.get(blk + "mlp.gate.e_score_correction_bias").to(torch.float32)
                if reader.has(blk + "mlp.gate.e_score_correction_bias")
                else torch.zeros(n_exp, dtype=torch.float32)
            )
            yield blk + "mlp.gate.e_score_correction_bias", bias_t, "f32"
            if sh_inter > 0:
                bump(dense_fmt, 3)
                yield blk + "mlp.shared_experts.gate_proj.weight", reader.get(blk + "mlp.shared_experts.gate_proj.weight"), dense_spec
                yield blk + "mlp.shared_experts.up_proj.weight", reader.get(blk + "mlp.shared_experts.up_proj.weight"), dense_spec
                yield blk + "mlp.shared_experts.down_proj.weight", reader.get(blk + "mlp.shared_experts.down_proj.weight"), dense_spec

            eb, eg, em5, eshvq = expert_spec
            edb, edg, edm5, edshvq = expert_down_spec
            step = max(1, expert_batch_size)
            for e0 in range(0, n_exp, step):
                e1 = min(n_exp, e0 + step)
                cnt = e1 - e0
                gu_list, dn_list = [], []
                for e in range(e0, e1):
                    ep = blk + f"mlp.experts.{e}."
                    gu_list.append(reader.get(ep + "gate_proj.weight"))
                    gu_list.append(reader.get(ep + "up_proj.weight"))
                    dn_list.append(reader.get(ep + "down_proj.weight"))
                gu_blocks, eff_eg = quant_any_blocks(
                    torch.cat(gu_list, dim=0), bits=eb, group=eg, is_m5=em5, is_shvq=eshvq
                )
                del gu_list
                gu_meta = _hq_meta(eb, eff_eg, em5, eshvq)
                gu_bbytes = any_block_bytes(eb, eff_eg, is_m5=(gu_meta is None), is_shvq=eshvq)
                gu_blocks = gu_blocks.reshape(2 * cnt, moe_inter, h // eff_eg, gu_bbytes)

                dn_blocks, eff_edg = quant_any_blocks(
                    torch.cat(dn_list, dim=0), bits=edb, group=edg, is_m5=edm5, is_shvq=edshvq
                )
                del dn_list
                dn_meta = _hq_meta(edb, eff_edg, edm5, edshvq)
                dn_bbytes = any_block_bytes(edb, eff_edg, is_m5=(dn_meta is None), is_shvq=edshvq)
                dn_blocks = dn_blocks.reshape(cnt, h, moe_inter // eff_edg, dn_bbytes)

                bump(expert_fmt, 2 * cnt)
                bump(expert_down_fmt, cnt)
                for i, e in enumerate(range(e0, e1)):
                    ep = blk + f"mlp.experts.{e}."
                    yield ep + "gate_proj.weight", gu_blocks[2 * i], ("prepacked_hq4", [moe_inter, h], gu_meta)
                    yield ep + "up_proj.weight", gu_blocks[2 * i + 1], ("prepacked_hq4", [moe_inter, h], gu_meta)
                    yield ep + "down_proj.weight", dn_blocks[i], ("prepacked_hq4", [h, moe_inter], dn_meta)
                del gu_blocks, dn_blocks
        else:
            bump(dense_fmt, 3)
            yield blk + "mlp.gate_proj.weight", reader.get(blk + "mlp.gate_proj.weight"), dense_spec
            yield blk + "mlp.up_proj.weight", reader.get(blk + "mlp.up_proj.weight"), dense_spec
            yield blk + "mlp.down_proj.weight", reader.get(blk + "mlp.down_proj.weight"), dense_spec

    def stream():
        bump("bf16")
        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight"), "bf16"
        yield from _yield_norm(reader, "model.norm.", bump)
        if not tie_emb or reader.has("lm_head.weight"):
            bump(dense_fmt)
            yield "lm_head.weight", reader.get("lm_head.weight"), dense_spec

        for l in range(num_layers):
            is_mla = _is_mla_layer(tc, l)
            has_dsa = is_mla and idx_h > 0 and (l not in dense_attn)
            yield from emit_layer(l, is_mla=is_mla, has_dsa=has_dsa, is_moe=(l >= first_k and n_exp > 0))

        if int(tc.get("num_nextn_predict_layers") or 0) >= 1:
            yield from _yield_norm(reader, mblk + "enorm.", bump)
            yield from _yield_norm(reader, mblk + "hnorm.", bump)
            yield from _yield_norm(reader, mblk + "shared_head.norm.", bump)
            bump(dense_fmt)
            yield mblk + "eh_proj.weight", reader.get(mblk + "eh_proj.weight"), dense_spec
            if not dedup_mtp_tie:
                bump("bf16")
                bump(dense_fmt)
                yield mblk + "embed_tokens.weight", reader.get(mblk + "embed_tokens.weight"), "bf16"
                yield mblk + "shared_head.head.weight", reader.get(mblk + "shared_head.head.weight"), dense_spec
            nextn_is_mla = not bool(tc.get("nextn_is_linear", False))
            nextn_sparse = nextn_is_mla and bool(tc.get("nextn_is_sparse", False)) and idx_h > 0
            yield from emit_layer(num_layers, is_mla=nextn_is_mla, has_dsa=nextn_sparse, is_moe=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            assets = staging if flat_root else staging / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, dict] = {}
            for src in sorted(source.iterdir()):
                if (
                    not src.is_file()
                    or src.name.startswith(".")
                    or src.name.endswith((".safetensors", ".index.json", ".pyc"))
                    or src.name == "crc32.txt"
                ):
                    continue
                shutil.copyfile(src, assets / src.name)
                manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
                if not flat_root and src.name in ("README.md", "LICENSE.txt"):
                    shutil.copyfile(src, staging / src.name)
            if embedded_tpl and not (assets / "chat_template.jinja").is_file():
                tpl_dst = assets / "chat_template.jinja"
                tpl_dst.write_text(embedded_tpl, encoding="utf-8")
                manifest["chat_template.jinja"] = {"sha256": sha256(tpl_dst), "bytes": tpl_dst.stat().st_size}
            manifest_path = assets / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))

            weights = staging / "model.qwp"
            pack_file(
                stream(),
                weights,
                {
                    "model_id": MODEL_ID.get(mt, "gigachat4"),
                    "geometry_digest": geo_digest,
                    "source": str(source),
                    "sha256_source": sha256_source_hex,
                    "source_revision": (
                        source.name
                        if source.parent.name == "snapshots"
                        and len(source.name) == 40
                        and all(c in "0123456789abcdef" for c in source.name)
                        else None
                    ),
                    "quant_recorded": quant_recorded,
                    "hq8_reference": None,
                    "assets_dir": "." if flat_root else "assets",
                    "assets_manifest_sha256": sha256(manifest_path),
                    "mtp_shared_embed_and_head": dedup_mtp_tie,
                },
            )
            audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            if destination.exists():
                raise FileExistsError(f"output appeared during quantization: {destination}")
            staging.rename(destination)
    finally:
        reader.close()
    print(f"packed quant={quant_recorded} stats={stats} -> {destination}", flush=True)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--quant", default="hq4_g32_m5")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--dense-quant", default=None)
    ap.add_argument("--expert-quant", default=None)
    ap.add_argument("--expert-down-quant", default=None)
    ap.add_argument("--flat-root", action="store_true")
    ap.add_argument("--raw-gdn-ab", action="store_true")
    args = ap.parse_args()
    quantize_gigachat4_model(
        args.model,
        args.out,
        flat_root=args.flat_root,
        quant=args.quant,
        group_size=args.group_size,
        dense_quant=args.dense_quant,
        expert_quant=args.expert_quant,
        expert_down_quant=args.expert_down_quant,
        raw_gdn_ab=args.raw_gdn_ab,
    )
