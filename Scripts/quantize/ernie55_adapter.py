"""ERNIE-5.5-Flash-Base (ernie5_5_moe) canonical QWP1 packer & QLSA0001 builder.
Fuses HCA/MLA-DSA projections, preserves exact FP32 routers/mHC/MultiMax/APE/sinks,
aliases shared-last-layer ECHO, and batch-encodes SHVQ/HQ expert slabs on Metal.
"""
from __future__ import annotations

import hashlib, json, os, shutil, sys, tempfile
from collections import OrderedDict
from pathlib import Path
import numpy as np

from hq4 import GROUP
from hq_blocks import VALID_GROUPS, dequant_fp8_blockwise
from pack import Q4G_ALIGNMENT, pack_file, sha256, synthesize_base_chat_template
from shvq import (any_block_bytes, format_any_quant_spec,
                  parse_any_quant_spec, quant_any_blocks)
from verify import audit_pack, read_header

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402

QLSA_MAGIC = b"QLSA0001"
QLSA_ALIGN = 64


class ShardReader:
    """Bounded-LRU safetensors slice reader with optional FP8 block dequant."""

    def __init__(self, source: Path, files: list[Path], fp8_block_size: int = 128, max_open: int = 8):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.source = source
        self.fp8_block_size = fp8_block_size
        self.max_open = max_open
        self._handles: OrderedDict[Path, object] = OrderedDict()
        self.tensor_to_file: dict[str, Path] = {}
        self.shapes: dict[str, list[int]] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    if name in self.tensor_to_file:
                        raise ValueError(f"duplicate source tensor: {name}")
                    self.tensor_to_file[name] = path
                    self.shapes[name] = list(sf.get_slice(name).get_shape())
        idx_path = source / "model.safetensors.index.json"
        if idx_path.is_file():
            wmap = json.loads(idx_path.read_text())["weight_map"]
            actual = {k: p.name for k, p in self.tensor_to_file.items()}
            if actual != wmap:
                raise ValueError("source shards do not match model.safetensors.index.json")

    def has(self, name: str) -> bool:
        return name in self.tensor_to_file

    def first_present(self, *names: str) -> str:
        for n in names:
            if n in self.tensor_to_file:
                return n
        raise KeyError(f"missing required tensor; tried {names}")

    def _handle(self, path: Path):
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        while len(self._handles) >= self.max_open:
            self._handles.popitem(last=False)
        sf = self._safe_open(path, framework="pt", device="cpu")
        self._handles[path] = sf
        return sf

    def get(self, name: str):
        path = self.tensor_to_file[name]
        sf = self._handle(path)
        t = sf.get_tensor(name)
        sinv = f"{name}_scale_inv"
        if sinv in self.tensor_to_file:
            s = self._handle(self.tensor_to_file[sinv]).get_tensor(sinv)
            t = dequant_fp8_blockwise(t, s, block_size=self.fp8_block_size)
        return t

    def get_first(self, *names: str):
        return self.get(self.first_present(*names))

    def close(self) -> None:
        self._handles.clear()


def _hq_meta(bits: int, group: int, is_m5: bool, is_shvq: bool) -> dict | None:
    use_m5 = (not is_shvq) and (bits == 4 and group == 32 and is_m5)
    if is_shvq:
        return {"bits": bits, "group": group, "vq": "shvq1"}
    if not use_m5:
        return {"bits": bits, "group": group}
    return None


def _is_compressed_layer(tc: dict, layer: int) -> bool:
    ltypes = tc.get("layer_types")
    if isinstance(ltypes, list) and layer < len(ltypes):
        return ltypes[layer] == "compressed_attention"
    cratios = tc.get("compress_ratios")
    if isinstance(cratios, list) and layer < len(cratios):
        return int(cratios[layer]) > 0
    return False


def _is_moe_layer(tc: dict, layer: int) -> bool:
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    if n_exp <= 0:
        return False
    first_k = int(tc.get("first_k_dense_replace", 0))
    freq = max(1, int(tc.get("moe_layer_freq", 1)))
    return layer >= first_k and (layer % freq == 0)


def _compute_hca_svd_f16(wkv, comp_norm, nope_dim: int, index_dim: int) -> np.ndarray:
    import torch
    w = wkv.detach().to(torch.float64)[:nope_dim]
    if comp_norm is not None:
        g = comp_norm.detach().to(torch.float64)[:nope_dim].unsqueeze(1)
        w = w * g
    gram = (1.0 / float(max(1, w.shape[1]))) * (w @ w.T)
    gram = 0.5 * (gram + gram.T)
    _, eigvecs = torch.linalg.eigh(gram)
    top = eigvecs[:, -index_dim:].flip(dims=[1]).T.contiguous()
    arr = top.to(torch.float32).cpu().numpy()
    max_idx = np.argmax(np.abs(arr), axis=1)
    signs = np.where(arr[np.arange(index_dim), max_idx] < 0.0, -1.0, 1.0).astype(np.float32)
    arr = arr * signs[:, None]
    return np.ascontiguousarray(arr.astype(np.float16).view("<u2"))


def write_hca_qlsa(reader: ShardReader, tc: dict, out_path: Path,
                   model_id: str, geometry_digest: str) -> dict | None:
    num_layers = int(tc["num_hidden_layers"])
    hca_layers = [l for l in range(num_layers) if _is_compressed_layer(tc, l)]
    nope_dim = int(tc.get("qk_nope_head_dim", 0))
    if not hca_layers or nope_dim <= 0:
        return None
    index_dim = min(64, nope_dim)
    cratios = tc.get("compress_ratios") or []
    pos_ratios = [int(r) for r in cratios[:num_layers] if int(r) > 0]
    pool_size = pos_ratios[0] if pos_ratios else 128
    window_toks = int(tc.get("sliding_window") or 128)
    tensors_meta: dict[str, dict] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(QLSA_MAGIC)
        f.write(b"\0" * 56)
        for l in hca_layers:
            blk = f"model.layers.{l}."
            if reader.has(blk + "self_attn.fused_in_proj.weight"):
                q_rank = int(tc["q_lora_rank"])
                head_dim = int(tc["head_dim"])
                fused = reader.get(blk + "self_attn.fused_in_proj.weight")
                wkv = fused[q_rank + head_dim: q_rank + 2 * head_dim]
            else:
                wkv = reader.get_first(
                    blk + "self_attn.core_attention.compressor.linear_wkv.weight",
                    blk + "self_attn.compress_kv.wkv.weight",
                )
            cnorm = reader.get_first(
                blk + "self_attn.comp_norm.weight",
                blk + "self_attn.core_attention.compressor.norm.weight",
                blk + "self_attn.compress_kv.norm.weight",
            )
            basis_u16 = _compute_hca_svd_f16(wkv, cnorm, nope_dim=nope_dim, index_dim=index_dim)
            raw_bytes = basis_u16.tobytes()
            for suffix in ("wq_idx", "wk_idx"):
                off = f.tell()
                pad = (-off) % QLSA_ALIGN
                if pad:
                    f.write(b"\0" * pad)
                    off += pad
                tname = f"layers.{l}.{suffix}"
                tensors_meta[tname] = {
                    "shape": [index_dim, nope_dim],
                    "dtype": "f16",
                    "raw_off": off,
                    "n_bytes": len(raw_bytes),
                }
                f.write(raw_bytes)
        off = f.tell()
        pad = (-off) % QLSA_ALIGN
        if pad:
            f.write(b"\0" * pad)
        hdr_obj = {
            "provenance": {
                "format": "QLSA0001",
                "model_id": model_id,
                "geometry_digest": geometry_digest,
                "method": "weight_svd_v1",
                "index_dim": index_dim,
                "pool_size": pool_size,
                "cluster_block_size": 1024,
                "buckets_per_block": 16,
                "max_bucket_keys": 64,
                "recommended_budget": int(tc.get("index_topk", 2048)),
                "sink_tokens": pool_size,
                "window_tokens": window_toks,
            },
            "tensors": tensors_meta,
        }
        hdr_bytes = json.dumps(hdr_obj, sort_keys=True).encode("utf-8")
        f.write(hdr_bytes)
        f.seek(8)
        f.write(len(hdr_bytes).to_bytes(8, "little"))
    return {"sha256": sha256(out_path), "bytes": out_path.stat().st_size}


def _fold_mhc_sub(reader: ShardReader, blk: str, canonical: str, hf_sub: str, streams: int, h: int):
    import torch
    rows = 2 * streams + streams * streams
    width = streams * h
    if reader.has(blk + f"{canonical}.weight"):
        w = reader.get(blk + f"{canonical}.weight").to(torch.float32).reshape(rows, width)
        b = reader.get(blk + f"{canonical}.bias").to(torch.float32).reshape(rows)
        return w.contiguous(), b.contiguous()
    pfx = blk + hf_sub + "."
    mp = reader.get(pfx + "mapping_proj.weight").to(torch.float32).reshape(width, rows).T.contiguous()
    a_pre = float(reader.get(pfx + "alpha_pre").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_pre") else 1.0
    a_post = float(reader.get(pfx + "alpha_post").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_post") else 1.0
    a_res = float(reader.get(pfx + "alpha_res").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_res") else 1.0
    mp[:streams] *= a_pre
    mp[streams: 2 * streams] *= a_post
    mp[2 * streams:] *= a_res
    b = reader.get(pfx + "bias").to(torch.float32).reshape(rows).contiguous()
    return mp, b


def _fold_exit_head(reader: ShardReader, prefix: str, alt_prefix: str, streams: int, h: int):
    import torch
    width = streams * h
    if reader.has(prefix + ".weight"):
        w = reader.get(prefix + ".weight").to(torch.float32).reshape(streams, width)
        b = reader.get(prefix + ".bias").to(torch.float32).reshape(streams)
        return w.contiguous(), b.contiguous()
    fn = reader.get_first(prefix + ".hc_head_fn", alt_prefix + "hc_head_fn").to(torch.float32)
    sc_name = prefix + ".hc_head_scale" if reader.has(prefix + ".hc_head_scale") else (alt_prefix + "hc_head_scale")
    sc = float(reader.get(sc_name).to(torch.float32).reshape(-1)[0].item()) if reader.has(sc_name) else 1.0
    bs = reader.get_first(prefix + ".hc_head_base", alt_prefix + "hc_head_base").to(torch.float32).reshape(streams)
    w = (fn.reshape(width, streams).T.contiguous() * sc)
    return w, bs.contiguous()


def _fold_vha(reader: ShardReader, blk: str, q_heads: int):
    import torch
    if reader.has(blk + "self_attn.vha_mix.weight"):
        return reader.get(blk + "self_attn.vha_mix.weight").to(torch.float32).reshape(q_heads, q_heads).contiguous()
    u = reader.get(blk + "self_attn.vha_postmix_U").to(torch.float32)
    v = reader.get(blk + "self_attn.vha_postmix_V").to(torch.float32)
    eye = torch.eye(q_heads, dtype=torch.float32)
    return (eye + (v @ u.T)).contiguous()


def build_expected_shapes(tc: dict) -> dict[str, list[int]]:
    h = int(tc["hidden_size"])
    v = int(tc["vocab_size"])
    num_layers = int(tc["num_hidden_layers"])
    inter = int(tc["intermediate_size"])
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size", inter))
    lat_size = int(tc.get("moe_latent_size") or 0)
    routed_dim = lat_size if lat_size > 0 else h
    n_shared = int(tc.get("n_shared_experts") or 0)
    sh_inter = int(tc.get("shared_expert_intermediate_size") or (n_shared * moe_inter))
    split_router = bool(tc.get("moe_split_feature_routing", False))
    learn_scale = bool(tc.get("routed_scaling_factor_learnable", False))
    has_mhc = bool(tc.get("enable_hyper_connections", False))
    streams = int(tc.get("hc_mult", 4))
    mhc_rows = 2 * streams + streams * streams
    use_vha = bool(tc.get("use_vha_attention", False))
    has_sink = bool(tc.get("add_full_attention_sink_bias", False))
    has_ogate = bool(tc.get("use_gated_attn", False))

    exp: dict[str, list[int]] = {
        "model.embed_tokens.weight": [v, h],
        "model.norm.weight": [h],
    }
    if not bool(tc.get("tie_word_embeddings", False)):
        exp["lm_head.weight"] = [v, h]
    if has_mhc:
        exp["model.mhc_contract.weight"] = [streams, streams * h]
        exp["model.mhc_contract.bias"] = [streams]
    if tc.get("multimax") == "lm_head":
        exp["lm_head.multimax"] = [8]

    for l in range(num_layers):
        blk = f"model.layers.{l}."
        exp[blk + "input_layernorm.weight"] = [h]
        exp[blk + "post_attention_layernorm.weight"] = [h]
        if has_mhc:
            exp[blk + "mhc_attn.weight"] = [mhc_rows, streams * h]
            exp[blk + "mhc_attn.bias"] = [mhc_rows]
            exp[blk + "mhc_ffn.weight"] = [mhc_rows, streams * h]
            exp[blk + "mhc_ffn.bias"] = [mhc_rows]
        if _is_compressed_layer(tc, l):
            q_heads = int(tc["num_attention_heads"])
            head_dim = int(tc["head_dim"])
            q_rank = int(tc["q_lora_rank"])
            o_groups = max(1, int(tc.get("o_groups", 1)))
            o_lora = int(tc["o_lora_rank"])
            comp_ratio = int(tc["compress_ratios"][l])
            exp[blk + "self_attn.fused_in_proj.weight"] = [q_rank + 3 * head_dim, h]
            exp[blk + "self_attn.q_norm.weight"] = [q_rank]
            exp[blk + "self_attn.kv_norm.weight"] = [head_dim]
            exp[blk + "self_attn.comp_norm.weight"] = [head_dim]
            exp[blk + "self_attn.compressor.ape"] = [comp_ratio, head_dim]
            exp[blk + "self_attn.q_up_gate_proj.weight"] = [q_heads * head_dim + o_groups * o_lora, q_rank]
            exp[blk + "self_attn.wo_a.weight"] = [o_groups * o_lora, (q_heads // o_groups) * head_dim]
            exp[blk + "self_attn.o_proj.weight"] = [h, o_groups * o_lora]
            if use_vha:
                exp[blk + "self_attn.vha_mix.weight"] = [q_heads, q_heads]
            if has_sink:
                exp[blk + "self_attn.attn_sink"] = [q_heads]
        else:
            q_heads = int(tc.get("hybrid_mla_num_attention_heads", tc["num_attention_heads"]))
            q_rank = int(tc.get("hybrid_mla_q_lora_rank", tc["q_lora_rank"]))
            kv_rank = int(tc.get("hybrid_mla_kv_lora_rank", tc["kv_lora_rank"]))
            nope_d = int(tc.get("hybrid_mla_qk_nope_head_dim", tc["qk_nope_head_dim"]))
            rope_d = int(tc.get("hybrid_mla_qk_rope_head_dim", tc["qk_rope_head_dim"]))
            v_d = int(tc.get("hybrid_mla_v_head_dim", tc["v_head_dim"]))
            idx_h = int(tc.get("index_n_heads", q_heads))
            idx_d = int(tc.get("index_head_dim", 128))
            q_ch = q_heads * (nope_d + rope_d + (v_d if has_ogate else 0))
            exp[blk + "self_attn.q_a_proj.weight"] = [q_rank, h]
            exp[blk + "self_attn.kv_a_proj_with_mqa.weight"] = [kv_rank + rope_d, h]
            exp[blk + "self_attn.indexer.wk.weight"] = [idx_d, h]
            exp[blk + "self_attn.q_a_layernorm.weight"] = [q_rank]
            exp[blk + "self_attn.kv_a_layernorm.weight"] = [kv_rank]
            exp[blk + "self_attn.indexer.k_norm.weight"] = [idx_d]
            exp[blk + "self_attn.indexer.k_norm.bias"] = [idx_d]
            exp[blk + "self_attn.indexer.weights_proj.weight"] = [idx_h, h]
            exp[blk + "self_attn.q_b_gate_idx_proj.weight"] = [q_ch + idx_h * idx_d, q_rank]
            exp[blk + "self_attn.kv_b_proj.weight"] = [q_heads * (nope_d + v_d), kv_rank]
            exp[blk + "self_attn.o_proj.weight"] = [h, q_heads * v_d]
            if use_vha:
                exp[blk + "self_attn.vha_mix.weight"] = [q_heads, q_heads]
            if has_sink:
                exp[blk + "self_attn.attn_sink"] = [q_heads]

        if _is_moe_layer(tc, l):
            r_rows = (2 * n_exp) if split_router else n_exp
            exp[blk + "mlp.gate.weight"] = [r_rows, h]
            exp[blk + "mlp.gate.e_score_correction_bias"] = [n_exp]
            if learn_scale:
                exp[blk + "mlp.routed_scaling_factor_param"] = [n_exp]
            if lat_size > 0:
                exp[blk + "mlp.fc1.weight"] = [lat_size, h]
                exp[blk + "mlp.latent_norm.weight"] = [lat_size]
                exp[blk + "mlp.fc2.weight"] = [h, lat_size]
            if sh_inter > 0:
                exp[blk + "mlp.shared_experts.gate_proj.weight"] = [sh_inter, h]
                exp[blk + "mlp.shared_experts.up_proj.weight"] = [sh_inter, h]
                exp[blk + "mlp.shared_experts.down_proj.weight"] = [h, sh_inter]
            for e in range(n_exp):
                exp[blk + f"mlp.experts.{e}.gate_proj.weight"] = [moe_inter, routed_dim]
                exp[blk + f"mlp.experts.{e}.up_proj.weight"] = [moe_inter, routed_dim]
                exp[blk + f"mlp.experts.{e}.down_proj.weight"] = [routed_dim, moe_inter]
        else:
            exp[blk + "mlp.gate_proj.weight"] = [inter, h]
            exp[blk + "mlp.up_proj.weight"] = [inter, h]
            exp[blk + "mlp.down_proj.weight"] = [h, inter]

    if int(tc.get("num_nextn_predict_layers") or 0) >= 1 and bool(tc.get("mtp_shared_last_layer", False)):
        eblk = f"model.layers.{num_layers}."
        exp[eblk + "enorm.weight"] = [h]
        exp[eblk + "hnorm.weight"] = [h]
        exp[eblk + "e_proj.weight"] = [h, h]
        exp[eblk + "h_proj.weight"] = [h, h]
        exp[eblk + "mhc_contract.weight"] = [streams, streams * h]
        exp[eblk + "mhc_contract.bias"] = [streams]
        exp[eblk + "shared_head.norm.weight"] = [h]
        if n_exp > 0:
            exp[eblk + "mlp.gate.e_score_correction_bias"] = [n_exp]
    return exp


def quantize_ernie55_model(
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
):
    import torch
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc = config["text_config"] if "text_config" in config else config
    qcfg = config.get("quantization_config")
    fp8_block_size = 128
    if qcfg is not None:
        if not isinstance(qcfg, dict) or qcfg.get("quant_method") != "fp8":
            raise ValueError("REFUSED: source is already quantized; use the BF16/FP8 checkpoint")
        wbs = qcfg.get("weight_block_size")
        if isinstance(wbs, list) and len(wbs) == 2 and wbs[0] == wbs[1]:
            fp8_block_size = int(wbs[0])

    if not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    has_jinja_file = (source / "chat_template.jinja").is_file()
    embedded_tpl = None
    if not has_jinja_file:
        tok_cfg_p = source / "tokenizer_config.json"
        if tok_cfg_p.is_file():
            tok_cfg = json.loads(tok_cfg_p.read_text())
            if isinstance(tok_cfg.get("chat_template"), str):
                embedded_tpl = tok_cfg["chat_template"]
        if not embedded_tpl:
            embedded_tpl = synthesize_base_chat_template(source)

    if dense_quant is None and expert_quant is None and expert_down_quant is None and quant == "hq4_g32_m5":
        dense_quant = "hq8_g32"
        expert_quant = "shvq15_g64"
        expert_down_quant = "shvq3_g64"
    dense_spec = parse_any_quant_spec(dense_quant or quant, default_group=group_size)
    expert_spec = parse_any_quant_spec(expert_quant or quant, default_group=group_size)
    expert_down_spec = parse_any_quant_spec(
        expert_down_quant or expert_quant or quant, default_group=group_size)
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
    expected_shapes = build_expected_shapes(tc)

    sha256_source_hex = None
    if reuse_source_sha_from is not None:
        rq = Path(reuse_source_sha_from) / "model.qwp"
        if rq.is_file():
            rp = read_header(rq)[0].get("provenance", {})
            if rp.get("source") == str(source) and rp.get("sha256_source"):
                sha256_source_hex = rp["sha256_source"]
    if sha256_source_hex is None:
        h_src = hashlib.sha256()
        for idx, path in enumerate(files, 1):
            if idx == 1 or idx % 16 == 0 or idx == len(files):
                print(f"hashing shard {idx}/{len(files)} ({path.name})", flush=True)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(16 << 20), b""):
                    h_src.update(chunk)
        sha256_source_hex = h_src.hexdigest()

    model_id = MODEL_ID[tc["model_type"]]
    geo_digest = hashlib.sha256(geo_v2(tc).encode()).hexdigest()
    identity = {"model_id": model_id, "geometry_digest": geo_digest}
    stats: dict[str, int] = {"bf16": 0, "f32": 0}

    def bump(k: str, n: int = 1) -> None:
        stats[k] = stats.get(k, 0) + n

    def stream():
        h = int(tc["hidden_size"])
        num_layers = int(tc["num_hidden_layers"])
        n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
        moe_inter = int(tc.get("moe_intermediate_size", tc["intermediate_size"]))
        lat_size = int(tc.get("moe_latent_size") or 0)
        routed_dim = lat_size if lat_size > 0 else h
        n_shared = int(tc.get("n_shared_experts") or 0)
        sh_inter = int(tc.get("shared_expert_intermediate_size") or (n_shared * moe_inter))
        split_router = bool(tc.get("moe_split_feature_routing", False))
        learn_scale = bool(tc.get("routed_scaling_factor_learnable", False))
        has_mhc = bool(tc.get("enable_hyper_connections", False))
        streams = int(tc.get("hc_mult", 4))
        use_vha = bool(tc.get("use_vha_attention", False))
        has_sink = bool(tc.get("add_full_attention_sink_bias", False))
        has_ogate = bool(tc.get("use_gated_attn", False))

        bump("bf16", 2)
        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight"), "bf16"
        yield "model.norm.weight", reader.get("model.norm.weight"), "bf16"
        if not bool(tc.get("tie_word_embeddings", False)):
            bump(dense_fmt)
            yield "lm_head.weight", reader.get("lm_head.weight"), dense_spec
        if has_mhc:
            mw, mb = _fold_exit_head(reader, "model.mhc_contract", "model.mhc_contract.", streams, h)
            bump("f32", 2)
            yield "model.mhc_contract.weight", mw, "f32"
            yield "model.mhc_contract.bias", mb, "f32"
        if tc.get("multimax") == "lm_head":
            if reader.has("lm_head.multimax"):
                mm = reader.get("lm_head.multimax").to(torch.float32).reshape(8)
            else:
                mr = reader.get("model.lm_head.multimax_ranges").to(torch.float32).reshape(4)
                mt = reader.get("model.lm_head.multimax_ts").to(torch.float32).reshape(4)
                mm = torch.cat([mr, mt], dim=0)
            bump("f32")
            yield "lm_head.multimax", mm, "f32"

        eb, eg, em5, eshvq = expert_spec
        edb, edg, edm5, edshvq = expert_down_spec

        for l in range(num_layers):
            print(f"quantizing layer {l}/{num_layers - 1}", flush=True)
            blk = f"model.layers.{l}."
            bump("bf16", 2)
            yield blk + "input_layernorm.weight", reader.get(blk + "input_layernorm.weight"), "bf16"
            yield blk + "post_attention_layernorm.weight", reader.get(blk + "post_attention_layernorm.weight"), "bf16"
            if has_mhc:
                aw, ab = _fold_mhc_sub(reader, blk, "mhc_attn", "self_attention_hyper_connection", streams, h)
                fw, fb = _fold_mhc_sub(reader, blk, "mhc_ffn", "mlp_hyper_connection", streams, h)
                bump("f32", 4)
                yield blk + "mhc_attn.weight", aw, "f32"
                yield blk + "mhc_attn.bias", ab, "f32"
                yield blk + "mhc_ffn.weight", fw, "f32"
                yield blk + "mhc_ffn.bias", fb, "f32"

            if _is_compressed_layer(tc, l):
                q_heads = int(tc["num_attention_heads"])
                head_dim = int(tc["head_dim"])
                if reader.has(blk + "self_attn.fused_in_proj.weight"):
                    fused_in = reader.get(blk + "self_attn.fused_in_proj.weight")
                else:
                    fused_in = torch.cat([
                        reader.get(blk + "self_attn.linear_q_down_proj.weight"),
                        reader.get(blk + "self_attn.linear_kv_proj.weight"),
                        reader.get_first(
                            blk + "self_attn.core_attention.compressor.linear_wkv.weight",
                            blk + "self_attn.compress_kv.wkv.weight"),
                        reader.get_first(
                            blk + "self_attn.core_attention.compressor.linear_wgate.weight",
                            blk + "self_attn.compress_kv.wgate.weight"),
                    ], dim=0)
                bump(dense_fmt)
                yield blk + "self_attn.fused_in_proj.weight", fused_in, dense_spec
                del fused_in

                bump("bf16", 3)
                yield blk + "self_attn.q_norm.weight", reader.get_first(
                    blk + "self_attn.q_norm.weight", blk + "self_attn.q_layernorm.weight"), "bf16"
                yield blk + "self_attn.kv_norm.weight", reader.get_first(
                    blk + "self_attn.kv_norm.weight", blk + "self_attn.kv_layernorm.weight"), "bf16"
                yield blk + "self_attn.comp_norm.weight", reader.get_first(
                    blk + "self_attn.comp_norm.weight",
                    blk + "self_attn.core_attention.compressor.norm.weight",
                    blk + "self_attn.compress_kv.norm.weight"), "bf16"

                bump("f32")
                yield blk + "self_attn.compressor.ape", reader.get_first(
                    blk + "self_attn.compressor.ape",
                    blk + "self_attn.core_attention.compressor.ape"), "f32"

                if reader.has(blk + "self_attn.q_up_gate_proj.weight"):
                    qup_gate = reader.get(blk + "self_attn.q_up_gate_proj.weight")
                else:
                    qup_gate = torch.cat([
                        reader.get(blk + "self_attn.linear_q_up_proj.weight"),
                        reader.get(blk + "self_attn.gate_proj.weight"),
                    ], dim=0)
                bump(dense_fmt, 3)
                yield blk + "self_attn.q_up_gate_proj.weight", qup_gate, dense_spec
                del qup_gate
                yield blk + "self_attn.wo_a.weight", reader.get_first(
                    blk + "self_attn.wo_a.weight", blk + "self_attn.linear_o_group_proj"), dense_spec
                yield blk + "self_attn.o_proj.weight", reader.get(blk + "self_attn.o_proj.weight"), dense_spec
                if use_vha:
                    bump("f32")
                    yield blk + "self_attn.vha_mix.weight", _fold_vha(reader, blk, q_heads), "f32"
                if has_sink:
                    bump("f32")
                    yield blk + "self_attn.attn_sink", reader.get_first(
                        blk + "self_attn.attn_sink",
                        blk + "self_attn.core_attention.attn_sink",
                        blk + "self_attn.core_attention.softmax_offset"), "f32"
            else:
                q_heads = int(tc.get("hybrid_mla_num_attention_heads", tc["num_attention_heads"]))
                q_rank = int(tc.get("hybrid_mla_q_lora_rank", tc["q_lora_rank"]))
                nope_d = int(tc.get("hybrid_mla_qk_nope_head_dim", tc["qk_nope_head_dim"]))
                rope_d = int(tc.get("hybrid_mla_qk_rope_head_dim", tc["qk_rope_head_dim"]))
                idx_pfx = (blk + "self_attn.indexer.") if reader.has(blk + "self_attn.indexer.wk.weight") else (
                    blk + "self_attn.core_attention.indexer.")

                bump(dense_fmt, 3)
                yield blk + "self_attn.q_a_proj.weight", reader.get(blk + "self_attn.q_a_proj.weight"), dense_spec
                yield blk + "self_attn.kv_a_proj_with_mqa.weight", reader.get(
                    blk + "self_attn.kv_a_proj_with_mqa.weight"), dense_spec
                yield blk + "self_attn.indexer.wk.weight", reader.get(idx_pfx + "wk.weight"), dense_spec

                bump("bf16", 4)
                yield blk + "self_attn.q_a_layernorm.weight", reader.get(
                    blk + "self_attn.q_a_layernorm.weight"), "bf16"
                yield blk + "self_attn.kv_a_layernorm.weight", reader.get(
                    blk + "self_attn.kv_a_layernorm.weight"), "bf16"
                yield blk + "self_attn.indexer.k_norm.weight", reader.get(idx_pfx + "k_norm.weight"), "bf16"
                yield blk + "self_attn.indexer.k_norm.bias", reader.get(idx_pfx + "k_norm.bias"), "bf16"

                bump("f32")
                yield blk + "self_attn.indexer.weights_proj.weight", reader.get(
                    idx_pfx + "weights_proj.weight"), "f32"

                if reader.has(blk + "self_attn.q_b_gate_idx_proj.weight"):
                    qb_fused = reader.get(blk + "self_attn.q_b_gate_idx_proj.weight")
                else:
                    if reader.has(blk + "self_attn.q_b_proj.weight"):
                        qb_raw = reader.get(blk + "self_attn.q_b_proj.weight").view(q_heads, nope_d + rope_d, q_rank)
                        q_nope = qb_raw[:, :nope_d, :].reshape(q_heads * nope_d, q_rank)
                        q_rope = qb_raw[:, nope_d:, :].reshape(q_heads * rope_d, q_rank)
                    else:
                        q_nope = reader.get(blk + "self_attn.q_b_nope_proj.weight")
                        q_rope = reader.get(blk + "self_attn.q_b_rope_proj.weight")
                    parts = [q_nope, q_rope]
                    if has_ogate:
                        parts.append(reader.get(blk + "self_attn.gate_proj.weight"))
                    parts.append(reader.get(idx_pfx + "wq_b.weight"))
                    qb_fused = torch.cat(parts, dim=0)
                bump(dense_fmt, 3)
                yield blk + "self_attn.q_b_gate_idx_proj.weight", qb_fused, dense_spec
                del qb_fused
                yield blk + "self_attn.kv_b_proj.weight", reader.get(blk + "self_attn.kv_b_proj.weight"), dense_spec
                yield blk + "self_attn.o_proj.weight", reader.get(blk + "self_attn.o_proj.weight"), dense_spec
                if use_vha:
                    bump("f32")
                    yield blk + "self_attn.vha_mix.weight", _fold_vha(reader, blk, q_heads), "f32"
                if has_sink:
                    bump("f32")
                    yield blk + "self_attn.attn_sink", reader.get_first(
                        blk + "self_attn.attn_sink",
                        blk + "self_attn.core_attention.softmax_offset"), "f32"

            if _is_moe_layer(tc, l):
                if reader.has(blk + "mlp.gate.weight"):
                    r_w = reader.get(blk + "mlp.gate.weight").to(torch.float32)
                elif split_router:
                    r_w = torch.cat([
                        reader.get(blk + "block_sparse_moe.gate.weight").to(torch.float32),
                        reader.get(blk + "block_sparse_moe.gate.weight_1").to(torch.float32),
                    ], dim=0)
                else:
                    r_w = reader.get(blk + "block_sparse_moe.gate.weight").to(torch.float32)
                bump("f32", 2)
                yield blk + "mlp.gate.weight", r_w, "f32"
                yield blk + "mlp.gate.e_score_correction_bias", reader.get_first(
                    blk + "mlp.gate.e_score_correction_bias",
                    blk + "block_sparse_moe.e_score_correction_bias"), "f32"
                if learn_scale:
                    bump("f32")
                    yield blk + "mlp.routed_scaling_factor_param", reader.get_first(
                        blk + "mlp.routed_scaling_factor_param",
                        blk + "block_sparse_moe.gate.routed_scaling_factor_param"), "f32"
                if lat_size > 0:
                    bump(dense_fmt, 2)
                    bump("bf16")
                    yield blk + "mlp.fc1.weight", reader.get_first(
                        blk + "mlp.fc1.weight", blk + "block_sparse_moe.fc1_latent_proj.weight"), dense_spec
                    yield blk + "mlp.latent_norm.weight", reader.get_first(
                        blk + "mlp.latent_norm.weight", blk + "block_sparse_moe.latent_norm.weight"), "bf16"
                    yield blk + "mlp.fc2.weight", reader.get_first(
                        blk + "mlp.fc2.weight", blk + "block_sparse_moe.fc2_latent_proj.weight"), dense_spec
                if sh_inter > 0:
                    bump(dense_fmt, 3)
                    yield blk + "mlp.shared_experts.gate_proj.weight", reader.get_first(
                        blk + "mlp.shared_experts.gate_proj.weight",
                        blk + "block_sparse_moe.shared_experts.w1.weight"), dense_spec
                    yield blk + "mlp.shared_experts.up_proj.weight", reader.get_first(
                        blk + "mlp.shared_experts.up_proj.weight",
                        blk + "block_sparse_moe.shared_experts.w3.weight"), dense_spec
                    yield blk + "mlp.shared_experts.down_proj.weight", reader.get_first(
                        blk + "mlp.shared_experts.down_proj.weight",
                        blk + "block_sparse_moe.shared_experts.w2.weight"), dense_spec

                use_bs = not reader.has(blk + "mlp.experts.0.gate_proj.weight")
                step = max(1, expert_batch_size)
                for e0 in range(0, n_exp, step):
                    e1 = min(n_exp, e0 + step)
                    cnt = e1 - e0
                    gu_list, dn_list = [], []
                    for e in range(e0, e1):
                        if use_bs:
                            epfx = blk + f"block_sparse_moe.experts.{e}."
                            gu_list.append(reader.get(epfx + "w1.weight"))
                            gu_list.append(reader.get(epfx + "w3.weight"))
                            dn_list.append(reader.get(epfx + "w2.weight"))
                        else:
                            epfx = blk + f"mlp.experts.{e}."
                            gu_list.append(reader.get(epfx + "gate_proj.weight"))
                            gu_list.append(reader.get(epfx + "up_proj.weight"))
                            dn_list.append(reader.get(epfx + "down_proj.weight"))
                    gu_cat = torch.cat(gu_list, dim=0)
                    del gu_list
                    gu_blocks, eff_eg = quant_any_blocks(
                        gu_cat, bits=eb, group=eg, is_m5=em5, is_shvq=eshvq)
                    del gu_cat
                    gu_meta = _hq_meta(eb, eff_eg, em5, eshvq)
                    gu_bbytes = any_block_bytes(
                        eb, eff_eg, is_m5=(gu_meta is None), is_shvq=eshvq)
                    gu_blocks = gu_blocks.reshape(2 * cnt, moe_inter, routed_dim // eff_eg, gu_bbytes)

                    dn_cat = torch.cat(dn_list, dim=0)
                    del dn_list
                    dn_blocks, eff_edg = quant_any_blocks(
                        dn_cat, bits=edb, group=edg, is_m5=edm5, is_shvq=edshvq)
                    del dn_cat
                    dn_meta = _hq_meta(edb, eff_edg, edm5, edshvq)
                    dn_bbytes = any_block_bytes(
                        edb, eff_edg, is_m5=(dn_meta is None), is_shvq=edshvq)
                    dn_blocks = dn_blocks.reshape(cnt, routed_dim, moe_inter // eff_edg, dn_bbytes)

                    bump(expert_fmt, 2 * cnt)
                    bump(expert_down_fmt, cnt)
                    for i, e in enumerate(range(e0, e1)):
                        ep = blk + f"mlp.experts.{e}."
                        yield ep + "gate_proj.weight", gu_blocks[2 * i], ("prepacked_hq4", [moe_inter, routed_dim], gu_meta)
                        yield ep + "up_proj.weight", gu_blocks[2 * i + 1], ("prepacked_hq4", [moe_inter, routed_dim], gu_meta)
                        yield ep + "down_proj.weight", dn_blocks[i], ("prepacked_hq4", [routed_dim, moe_inter], dn_meta)
                    del gu_blocks, dn_blocks
            else:
                bump(dense_fmt, 3)
                yield blk + "mlp.gate_proj.weight", reader.get(blk + "mlp.gate_proj.weight"), dense_spec
                yield blk + "mlp.up_proj.weight", reader.get(blk + "mlp.up_proj.weight"), dense_spec
                yield blk + "mlp.down_proj.weight", reader.get(blk + "mlp.down_proj.weight"), dense_spec

        if int(tc.get("num_nextn_predict_layers") or 0) >= 1 and bool(tc.get("mtp_shared_last_layer", False)):
            eblk = f"model.layers.{num_layers}."
            bump("bf16", 3)
            bump(dense_fmt, 2)
            bump("f32", 2)
            yield eblk + "enorm.weight", reader.get(eblk + "enorm.weight"), "bf16"
            yield eblk + "hnorm.weight", reader.get(eblk + "hnorm.weight"), "bf16"
            yield eblk + "e_proj.weight", reader.get(eblk + "e_proj.weight"), dense_spec
            yield eblk + "h_proj.weight", reader.get(eblk + "h_proj.weight"), dense_spec
            ew, eb_b = _fold_exit_head(reader, eblk + "mhc_contract", eblk, streams, h)
            yield eblk + "mhc_contract.weight", ew, "f32"
            yield eblk + "mhc_contract.bias", eb_b, "f32"
            yield eblk + "shared_head.norm.weight", reader.get(eblk + "shared_head.norm.weight"), "bf16"
            if n_exp > 0:
                bump("f32")
                yield eblk + "mlp.gate.e_score_correction_bias", reader.get_first(
                    eblk + "mlp.gate.e_score_correction_bias",
                    eblk + "block_sparse_moe.e_score_correction_bias",
                    f"model.layers.{num_layers - 1}.mlp.gate.e_score_correction_bias",
                    f"model.layers.{num_layers - 1}.block_sparse_moe.e_score_correction_bias"), "f32"

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            assets = staging if flat_root else staging / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, dict] = {}
            for src in sorted(source.iterdir()):
                if (not src.is_file() or src.name.startswith(".")
                        or src.name.endswith((".safetensors", ".index.json"))
                        or src.name == "crc32.txt"):
                    continue
                shutil.copyfile(src, assets / src.name)
                manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
            if embedded_tpl and not (assets / "chat_template.jinja").is_file():
                tpl_dst = assets / "chat_template.jinja"
                tpl_dst.write_text(embedded_tpl, encoding="utf-8")
                manifest["chat_template.jinja"] = {"sha256": sha256(tpl_dst), "bytes": tpl_dst.stat().st_size}
            qlsa_meta = write_hca_qlsa(reader, tc, assets / "model.qlsa", model_id, geo_digest)
            if qlsa_meta is not None:
                manifest["model.qlsa"] = qlsa_meta
            manifest_path = assets / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            weights = staging / "model.qwp"
            pack_file(stream(), weights, {
                **identity,
                "source": str(source),
                "sha256_source": sha256_source_hex,
                "source_revision": (source.name if source.parent.name == "snapshots"
                                    and len(source.name) == 40
                                    and all(c in "0123456789abcdef" for c in source.name)
                                    else None),
                "quant_recorded": quant_recorded,
                "hq8_reference": None,
                "assets_dir": "." if flat_root else "assets",
                "assets_manifest_sha256": sha256(manifest_path),
            })
            audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            if destination.exists():
                raise FileExistsError(f"output appeared during quantization: {destination}")
            staging.rename(destination)
    finally:
        reader.close()
    print(f"packed quant={quant_recorded} stats={stats} -> {destination}", flush=True)
