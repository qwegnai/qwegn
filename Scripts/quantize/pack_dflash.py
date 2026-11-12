"""DFlash 2 / DSpine drafter weight packer: BF16 -> HQ/SHVQ + BF16 QWP1 directory."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from pack import GROUP, pack_file, sha256
from shvq import format_any_quant_spec, parse_any_quant_spec
from verify import audit_pack

DFLASH_RAW_EXACT = {
    "hidden_norm.weight",
    "norm.weight",
    "candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
}

DSPINE_VERSION_TENSORS = {
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
}


def is_dflash_raw_plane(name: str) -> bool:
    """Return True for norm, dynamic-conv, selector, and DSpine bridge planes kept in BF16."""
    if name in DFLASH_RAW_EXACT or name.startswith("selected_prefix_bridge."):
        return True
    return any(
        token in name
        for token in (
            "layernorm",
            "q_norm",
            "k_norm",
            "base_kernel",
            "kernel_projection",
        )
    )


def quantize_dflash_model(
    source: Path, destination: Path, quant: str = "hq4_g32_m5"
) -> dict:
    """Pack a DFlash 2 or DSpine safetensors checkpoint into a QWP1 directory."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    spec = parse_any_quant_spec(quant)
    quant_canonical = format_any_quant_spec(*spec)
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    source = source.resolve(strict=True)
    destination = destination.absolute()
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
    if "DFlash2DraftModel" not in archs and "DFlashDraftModel" not in archs and "dflash_config" not in config:
        raise ValueError(f"not a DFlash2/DSpine DraftModel checkpoint: {archs}")

    shard_names, expected_shapes = {}, {}
    for path in files:
        with safe_open(path, framework="pt") as sf:
            shard_names[path] = list(sf.keys())
            for name in shard_names[path]:
                if name in expected_shapes:
                    raise ValueError(f"duplicate source tensor: {name}")
                if name in DSPINE_VERSION_TENSORS:
                    continue
                expected_shapes[name] = list(sf.get_slice(name).get_shape())

    h = hashlib.sha256()
    for path in files:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    sha256_source_hex = h.hexdigest()

    stats = {quant_canonical: 0, "bf16": 0}

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
                raw = is_dflash_raw_plane(name) or arr.ndim != 2 or arr.shape[1] % GROUP != 0
                stats["bf16" if raw else quant_canonical] += 1
                yield name, arr, (True if raw else spec)

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
            f"{config['num_attention_heads']}|{config['num_key_value_heads']}|"
            f"{config['head_dim']}|{config.get('block_size', dcfg.get('block_size', 16))}|"
            f"{code_size}|{rank}"
        )
        model_id = "dspine_qwen3_5_27b"
    else:
        geo_str = (
            f"qwg-dflash2-v1:{config['hidden_size']}|{config['vocab_size']}|"
            f"{config['intermediate_size']}|{config['num_hidden_layers']}|"
            f"{config['num_attention_heads']}|{config['num_key_value_heads']}|"
            f"{config['head_dim']}|{dcfg.get('block_size', 8)}|"
            f"{dcfg.get('selector_rank', 256)}|{dcfg.get('selector_top_k', 16)}"
        )
        model_id = "dflash2_qwen3_5_27b"

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
                "quant_recorded": quant_canonical,
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
    args = ap.parse_args()
    q_key = format_any_quant_spec(*parse_any_quant_spec(args.quant))
    res = quantize_dflash_model(Path(args.model), Path(args.out), quant=args.quant)
    print(f"packed {res[q_key]} {q_key.upper()} + {res['bf16']} BF16 tensors -> {args.out}")
