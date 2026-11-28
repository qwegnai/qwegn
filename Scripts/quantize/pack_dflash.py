"""DFlash 2 / DSpine drafter weight packer: BF16 -> HQ/SHVQ + BF16 QWP1 directory."""
import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from pack import GROUP, pack_file, resolve_snapshot_dir, sha256
from shvq import format_any_quant_spec, parse_any_quant_spec
from verify import audit_pack

DFLASH_RAW_EXACT = {
    "hidden_norm.weight",
    "norm.weight",
    "candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
}

DFLASH_MLP_SUFFIXES = (
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

DSPINE_VERSION_TENSORS = {
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
}


def is_dflash_raw_plane(name: str) -> bool:
    """Return True for norm, dynamic-conv, selector, and DSpine bridge planes kept in BF16."""
    if (
        name in DFLASH_RAW_EXACT
        or name.startswith("selected_prefix_bridge.")
        or name.startswith("candidate_selector.")
    ):
        return True
    return any(
        token in name
        for token in (
            "hidden_norm.",
            "norm.",
            "layernorm",
            "q_norm",
            "k_norm",
            "base_kernel",
            "kernel_projection",
            "gate_up_projection",
            "gate_down_projection",
        )
    )


def is_dflash_mlp_proj(name: str) -> bool:
    """Return True for drafter MLP linear projection weights (gate, up, down)."""
    return name in DFLASH_MLP_SUFFIXES or any(
        name.endswith("." + suffix) for suffix in DFLASH_MLP_SUFFIXES
    )


def derive_dflash_model_id(config: dict, source: Path, is_dspine: bool) -> str:
    """Derive provenance model_id with required dflash/dspine prefix from config or source."""
    prefix = "dspine_" if is_dspine else "dflash2_"
    dcfg = config.get("dflash_config") or {}
    raw_name = ""
    for candidate in (
        config.get("_name_or_path"),
        config.get("name_or_path"),
        dcfg.get("target_model_name") if isinstance(dcfg, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            raw_name = candidate.strip().rstrip("/").split("/")[-1]
            break
    if not raw_name:
        if source.parent.name == "snapshots" and source.parent.parent.name.startswith("models--"):
            parts = source.parent.parent.name.split("--", 2)
            raw_name = parts[2] if len(parts) >= 3 else parts[-1]
        elif (
            source.name in ("dflash_draft", "dflash2_draft", "draft")
            and source.parent.parent.name == "snapshots"
            and source.parent.parent.parent.name.startswith("models--")
        ):
            parts = source.parent.parent.parent.name.split("--", 2)
            raw_name = parts[2] if len(parts) >= 3 else parts[-1]
        elif source.name in ("dflash_draft", "dflash2_draft", "draft") and source.parent.name not in ("", "/", "."):
            raw_name = source.parent.name
        else:
            raw_name = source.name
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
    for suffix in ("_dflash2", "_dflash", "_dspine"):
        if slug.endswith(suffix) and len(slug) > len(suffix):
            slug = slug[: -len(suffix)].strip("_")
            break
    if slug in ("dflash2", "dflash", "dspine"):
        slug = ""
    if not slug:
        mt = str(config.get("model_type") or "drafter").lower()
        slug = re.sub(r"[^a-z0-9]+", "_", mt).strip("_") or "drafter"
    return f"{prefix}{slug}"


def quantize_dflash_model(
    source: Path,
    destination: Path,
    quant: str = "hq4_g32_m5",
    mlp_quant: str | None = None,
) -> dict:
    """Pack a DFlash 2 or DSpine safetensors checkpoint into a QWP1 directory."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    spec = parse_any_quant_spec(quant)
    quant_canonical = format_any_quant_spec(*spec)
    mlp_spec = parse_any_quant_spec(mlp_quant) if mlp_quant is not None else spec
    mlp_canonical = format_any_quant_spec(*mlp_spec)
    quant_recorded = (
        quant_canonical
        if mlp_canonical == quant_canonical
        else f"{quant_canonical}__mlp_{mlp_canonical}"
    )
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
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {source}")
    config = json.loads(config_path.read_text())
    archs = config.get("architectures") or []
    is_draft_arch = any("DraftModel" in str(a) or "DFlash" in str(a) for a in archs)
    if not is_draft_arch and "dflash_config" not in config:
        raise ValueError(f"not a DFlash2/DSpine DraftModel checkpoint: {archs}")

    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or q_heads)
    q_rank = int(config.get("q_lora_rank") or 0)
    nope_d = int(config.get("qk_nope_head_dim") or 0)
    rope_d = int(config.get("qk_rope_head_dim") or 0)
    split_mla_qb = q_rank > 0 and nope_d > 0 and rope_d > 0
    raw_hd = config.get("head_dim")
    if raw_hd is not None and not (nope_d > 0 and rope_d > 0 and int(raw_hd) == rope_d):
        resolved_hd = int(raw_hd)
    elif config.get("qk_head_dim"):
        resolved_hd = int(config["qk_head_dim"])
    elif nope_d + rope_d > 0:
        resolved_hd = nope_d + rope_d
    else:
        resolved_hd = int(config["hidden_size"]) // q_heads

    shard_names, expected_shapes = {}, {}
    seen_sources = set()
    for path in files:
        with safe_open(path, framework="pt") as sf:
            shard_names[path] = list(sf.keys())
            for name in shard_names[path]:
                if name in seen_sources:
                    raise ValueError(f"duplicate source tensor: {name}")
                seen_sources.add(name)
                if name in DSPINE_VERSION_TENSORS:
                    continue
                if split_mla_qb and name.endswith(".self_attn.q_b_proj.weight"):
                    pfx = name[: -len("q_b_proj.weight")]
                    expected_shapes[pfx + "q_b_nope_proj.weight"] = [q_heads * nope_d, q_rank]
                    expected_shapes[pfx + "q_b_rope_proj.weight"] = [q_heads * rope_d, q_rank]
                else:
                    expected_shapes[name] = list(sf.get_slice(name).get_shape())

    h = hashlib.sha256()
    for path in files:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    sha256_source_hex = h.hexdigest()

    stats = {quant_canonical: 0, "bf16": 0}
    if mlp_canonical != quant_canonical:
        stats[mlp_canonical] = 0

    def stream():
        for path in files:
            tensors = load_file(path)
            for name in shard_names[path]:
                arr = tensors.pop(name)
                if name in DSPINE_VERSION_TENSORS:
                    val = int(arr.item())
                    if val != 1:
                        raise ValueError(f"unsupported {name}={val} (expected 1)")
                    continue
                if split_mla_qb and name.endswith(".self_attn.q_b_proj.weight"):
                    pfx = name[: -len("q_b_proj.weight")]
                    viewed = arr.view(q_heads, nope_d + rope_d, q_rank)
                    qb_nope = viewed[:, :nope_d, :].reshape(q_heads * nope_d, q_rank).contiguous()
                    qb_rope = viewed[:, nope_d:, :].reshape(q_heads * rope_d, q_rank).contiguous()
                    stats[quant_canonical] += 2
                    yield pfx + "q_b_nope_proj.weight", qb_nope, spec
                    yield pfx + "q_b_rope_proj.weight", qb_rope, spec
                    continue
                raw = (
                    is_dflash_raw_plane(name)
                    or arr.ndim != 2
                    or arr.shape[0] < GROUP
                    or arr.shape[1] % GROUP != 0
                )
                if raw:
                    stats["bf16"] += 1
                    yield name, arr, True
                elif is_dflash_mlp_proj(name):
                    stats[mlp_canonical] += 1
                    yield name, arr, mlp_spec
                else:
                    stats[quant_canonical] += 1
                    yield name, arr, spec

    dcfg = config.get("dflash_config", {})
    bridge_cfg = dcfg.get("selected_prefix_code_bridge") or config.get("selected_prefix_code_bridge")
    is_dspine = isinstance(bridge_cfg, dict) or bool(bridge_cfg)
    if is_dspine:
        bdict = bridge_cfg if isinstance(bridge_cfg, dict) else {}
        code_size = bdict.get("code_size", config.get("selected_prefix_code_size", 1024))
        rank = bdict.get("rank", config.get("selected_prefix_rank", 512))
        geo_str = (
            f"qwg-dspine-v1:{config['hidden_size']}|{config['vocab_size']}|"
            f"{config['intermediate_size']}|{config['num_hidden_layers']}|"
            f"{q_heads}|{kv_heads}|"
            f"{resolved_hd}|{config.get('block_size', dcfg.get('block_size', 16))}|"
            f"{code_size}|{rank}"
        )
    else:
        geo_str = (
            f"qwg-dflash2-v1:{config['hidden_size']}|{config['vocab_size']}|"
            f"{config['intermediate_size']}|{config['num_hidden_layers']}|"
            f"{q_heads}|{kv_heads}|"
            f"{resolved_hd}|{dcfg.get('block_size', 8)}|"
            f"{dcfg.get('selector_rank', 256)}|{dcfg.get('selector_top_k', 16)}"
        )
    model_id = derive_dflash_model_id(config, source, is_dspine)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        staging = Path(temp) / "model"
        assets = staging / "assets"
        assets.mkdir(parents=True)
        manifest = {}
        for src in sorted(source.iterdir()):
            if (
                not src.is_file()
                or src.name.startswith(".")
                or src.name.endswith((".safetensors", ".index.json"))
            ):
                continue
            shutil.copyfile(src, assets / src.name)
            manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
        manifest_path = assets / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        weights = staging / "model.qwp"
        pack_file(
            stream(),
            weights,
            {
                "model_id": model_id,
                "geometry_hash": hashlib.sha256(geo_str.encode()).hexdigest(),
                "source": str(source),
                "sha256_source": sha256_source_hex,
                "quant_recorded": quant_recorded,
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256(manifest_path),
            },
        )
        audit_pack(weights, expected_shapes=expected_shapes)
        (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
        if destination.exists():
            raise FileExistsError(f"output appeared during quantization: {destination}")
        staging.rename(destination)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, metavar="DIR")
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument("--quant", default="hq4_g32_m5", metavar="SPEC")
    ap.add_argument("--mlp-quant", default=None, metavar="SPEC")
    args = ap.parse_args()
    res = quantize_dflash_model(
        Path(args.model), Path(args.out), quant=args.quant, mlp_quant=args.mlp_quant
    )
    q_parts = [f"{v} {k.upper()}" for k, v in res.items() if k != "bf16"]
    print(f"packed {' + '.join(q_parts)} + {res['bf16']} BF16 tensors -> {args.out}")
