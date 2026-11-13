"""QW weight packer: BF16/FP32 -> QWP1 model directory (HQ*, SHVQ*, PLE N-gram).
Layout (kernel order = storage order):
  hq4_g32_m5: 20 B Block_HQ4_G32 (f16 scale, f16 bias, uint32 qs[4]) = 5.0 bpw.
  3D expert stacks [E, N, K]: 16 KiB VM-page-aligned slab (experts=E, e_hq4, hq4_off).
"""
import hashlib, json, math, os, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from hq4 import (BLOCK_BYTES, GROUP, INV_SQRT32, dequant_hq4_g32_m5,
                 dequant_hq4_g32_m5_rot, fwht32_ortho, quant_hq4_g32_m5)
from hq_blocks import (VALID_GROUPS, dequant_fp8_blockwise, dequant_hq8_g32_m5,
                       dequant_hq8_g32_m5_rot, dequant_hq_blocks, format_quant_spec,
                       hq_block_bytes, parse_quant_spec, quant_hq8_g32_m5, quant_hq_blocks)
from pack_ngram import (HEAD_DIM, NGRAM_BIN_NAME, NGRAM_ROW_BYTES, PLE_LOCATION_NAME,
                        _NG_RPS, _NG_SECTOR, audit_ngram_bin, build_ngram_bin,
                        is_ple_ngram, scan_ple_and_router, write_ple_location_file)
from shvq import (any_block_bytes, dequant_any_blocks, format_any_quant_spec,
                  parse_any_quant_spec, quant_any_blocks)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402

HQ8_PAYLOAD_BYTES = 32
HQ8_SCALE_BYTES = 2
Q4G_ALIGNMENT = 16384
MAGIC = b"QWP1"

# Flat model root markers (resolved identically to QWModelRoot.swift).
FLAT_ROOT_MARKERS = ("config.json", NGRAM_BIN_NAME, PLE_LOCATION_NAME)


def resolve_assets_dir(root: Path) -> Path:
    """Resolve flat vs assets/ directory for a model root."""
    for name in FLAT_ROOT_MARKERS:
        if (root / name).is_file():
            return root
    return root / "assets"

HQ8_DENY = re.compile(r"\.experts\.|ngram_embedding|linear_attn\.in_proj_(qkv|z)\.|^mtp\.")
HQ8_DENSE_SUFFIXES = (
    ".linear_attn.out_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.indexer.index_qk_proj.weight",
    ".ple.key_proj.weight",
    ".ple.value_proj.weight",
)


def hq8_target(name: str) -> bool:
    return not HQ8_DENY.search(name) and name.endswith(HQ8_DENSE_SUFFIXES)

GEOMETRY = dict(hidden=5120, vocab=248320, intermediate=17408, num_layers=64,
                q_heads=24, kv_heads=4, head_dim=256, gdn_qkv_channels=10240,
                gdn_v_heads=48)


def geometry_hash_string():
    g = GEOMETRY
    return (f"qwg-geo-v1:{g['hidden']}|{g['vocab']}|{g['intermediate']}|"
            f"{g['num_layers']}|{g['q_heads']}|{g['kv_heads']}|"
            f"{g['head_dim']}|{g['gdn_qkv_channels']}|{g['gdn_v_heads']}")


def quant_q4g(w: np.ndarray) -> np.ndarray:
    return quant_hq4_g32_m5(w)


def dequant_q4g(blocks: np.ndarray) -> np.ndarray:
    return dequant_hq4_g32_m5(blocks)


def quant_q8_0(w: np.ndarray):
    N, K = w.shape
    g = w.reshape(N, K // GROUP, GROUP)
    amax = np.abs(g).max(-1, keepdims=True).clip(1e-8)
    scale = amax / 127.0
    q = np.rint(g / scale).clip(-127, 127).astype(np.int8)
    return q.reshape(N, K), scale[..., 0].astype(np.float16)


def dequant_q8_0(q, scale):
    N, K = q.shape
    return (q.astype(np.float32).reshape(N, K // GROUP, GROUP)
            * scale.astype(np.float32)[:, :, None]).reshape(N, K)


def _as_bf16_u16(arr) -> np.ndarray:
    import torch
    if isinstance(arr, torch.Tensor):
        return arr.detach().to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16).view(torch.uint16).numpy()


def _as_f32(arr) -> np.ndarray:
    import torch
    if isinstance(arr, torch.Tensor):
        return np.ascontiguousarray(arr.detach().to(torch.float32).cpu().numpy(), dtype="<f4")
    return np.ascontiguousarray(arr, dtype="<f4")


def expert_stack(name: str, shape) -> bool:
    if len(shape) != 3 or ".experts." not in name:
        return False
    E, N, K = shape
    if E <= 0 or N <= 0 or (N % 2) != 0 or K <= 0 or (K % GROUP) != 0:
        raise ValueError(
            f"{name}: expert stack shape {list(shape)} violates layout law "
            f"(want E>=1, even N, K%{GROUP}==0)")
    return True


def pack_file(tensors, out_path: Path, provenance: dict,
              raw=None, sidecar_dir=None, copy_entries=None):
    if hasattr(tensors, "items"):
        tensors = sorted(tensors.items())
    if raw is None:
        raw = []
    elif hasattr(raw, "items"):
        raw = sorted(raw.items())
    provenance = dict(provenance)
    provenance.setdefault("geometry_hash", hashlib.sha256(
        geometry_hash_string().encode()).hexdigest())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        hdr_off = 4
        f.write(b"\0" * 8)
        off = f.tell()
        pad = (-off) % 256
        f.write(b"\0" * pad)
        off += pad

        def pad_to(alignment: int):
            nonlocal off
            off = f.tell()
            p = (-off) % alignment
            f.write(b"\0" * p)
            off += p

        def pad256():
            pad_to(256)

        def pad_q4g():
            pad_to(Q4G_ALIGNMENT)

        for name, arr in raw:
            if name in index:
                raise ValueError(f"duplicate tensor: {name}")
            u16 = _as_bf16_u16(arr)
            entry = {
                "shape": list(arr.shape if hasattr(arr, "shape") else np.asarray(arr).shape),
                "kind": "bf16",
            }
            pad256()
            entry["raw_off"] = off
            f.write(memoryview(np.ascontiguousarray(u16)))
            pad256()
            index[name] = entry

        for item in tensors:
            name, arr, kind = (item if len(item) == 3 else (item[0], item[1], False))
            if name in index:
                raise ValueError(f"duplicate tensor: {name}")
            if isinstance(kind, tuple) and len(kind) == 3 and kind[0] == "prepacked_hq4":
                _, shape, hq_meta = kind
                pad_q4g()
                entry = {"shape": list(shape), "hq4_off": off}
                if hq_meta:
                    entry["hq"] = dict(hq_meta)
                f.write(memoryview(np.ascontiguousarray(arr)))
                pad_q4g()
                index[name] = entry
                continue
            shape = list(arr.shape if hasattr(arr, "shape") else np.asarray(arr).shape)
            if kind is True or kind == "bf16":
                u16 = _as_bf16_u16(arr)
                entry = {"shape": shape, "kind": "bf16"}
                pad256()
                entry["raw_off"] = off
                f.write(memoryview(np.ascontiguousarray(u16)))
                pad256()
                index[name] = entry
                continue
            if kind == "f32":
                f32 = _as_f32(arr)
                entry = {"shape": shape, "kind": "f32"}
                pad256()
                entry["raw_off"] = off
                f.write(memoryview(f32))
                pad256()
                index[name] = entry
                continue
            import torch as _t
            w = (arr if isinstance(arr, _t.Tensor)
                 else np.asarray(arr, dtype=np.float32))
            if kind == "hq8":
                if len(shape) != 2 or shape[1] % GROUP:
                    raise ValueError(f"{name}: hq8 requires 2-D K-aligned, got {shape}")
                qs, sc = quant_hq8_g32_m5(w)
                pad256()
                entry = {"shape": shape, "hq8_off": off}
                f.write(np.ascontiguousarray(qs).tobytes())
                pad256()
                entry["hq8_s_off"] = off
                f.write(np.ascontiguousarray(sc).tobytes())
                pad256()
                index[name] = entry
                continue
            is_expert = expert_stack(name, shape)
            if not is_expert and len(shape) != 2:
                raise ValueError(
                    f"{name}: unexpected rank-{len(shape)} shape {shape} — "
                    f"refuse to guess a layout")
            if kind is False or kind is None:
                bits, req_group, is_m5, is_shvq = (4, 32, True, False)
            elif isinstance(kind, str):
                bits, req_group, is_m5, is_shvq = parse_any_quant_spec(kind)
            elif len(kind) == 3:
                bits, req_group, is_m5 = kind
                is_shvq = False
            else:
                bits, req_group, is_m5, is_shvq = kind
            if is_expert:
                E3, N3, K3 = shape
                blocks, eff_group = quant_any_blocks(
                    w.reshape(E3 * N3, K3), bits=bits, group=req_group, is_m5=is_m5, is_shvq=is_shvq)
                use_m5 = (not is_shvq) and (bits == 4 and eff_group == 32 and is_m5)
                bbytes = any_block_bytes(bits, eff_group, is_m5=use_m5, is_shvq=is_shvq)
                e_hq4 = N3 * (K3 // eff_group) * bbytes
                pad_q4g()
                entry = {
                    "shape": shape, "experts": E3,
                    "e_hq4": e_hq4, "hq4_off": off,
                }
                if is_shvq:
                    entry["hq"] = {"bits": bits, "group": eff_group, "vq": "shvq1"}
                elif not use_m5:
                    entry["hq"] = {"bits": bits, "group": eff_group}
                f.write(memoryview(np.ascontiguousarray(blocks)))
                pad_q4g()
                index[name] = entry
                if sidecar_dir is not None:
                    sd = Path(sidecar_dir)
                    sd.mkdir(parents=True, exist_ok=True)
                    blocks_e = blocks.reshape(E3, N3, K3 // eff_group, bbytes)
                    for i in range(E3):
                        np.save(sd / f"{name}.e{i}.dq4.npy",
                                dequant_any_blocks(
                                    blocks_e[i], bits=bits, group=eff_group, is_m5=use_m5, is_shvq=is_shvq
                                ).astype(np.float32))
                continue
            N, K = w.shape
            assert K % GROUP == 0, f"{name}: K={K} not group-aligned"
            blocks, eff_group = quant_any_blocks(w, bits=bits, group=req_group, is_m5=is_m5, is_shvq=is_shvq)
            use_m5 = (not is_shvq) and (bits == 4 and eff_group == 32 and is_m5)
            pad_q4g()
            entry = {"shape": [N, K], "hq4_off": off}
            if is_shvq:
                entry["hq"] = {"bits": bits, "group": eff_group, "vq": "shvq1"}
            elif not use_m5:
                entry["hq"] = {"bits": bits, "group": eff_group}
            f.write(memoryview(np.ascontiguousarray(blocks)))
            pad_q4g()
            index[name] = entry

        if copy_entries is not None:
            src_path, ref_tensors = copy_entries
            chunk = 1 << 20
            with open(src_path, "rb") as rf:
                for name in sorted(ref_tensors):
                    ref = ref_tensors[name]
                    if "hq4_off" not in ref or "raw_off" in ref:
                        continue
                    if name in index:
                        if "hq8_off" not in index[name]:
                            raise ValueError(
                                f"copy tensor duplicates cooked tensor: {name}")
                        continue  # requantized to HQ8 in this pack
                    pad_q4g()
                    if "experts" in ref:
                        length = ref["experts"] * ref["e_hq4"]
                        entry = {"shape": ref["shape"], "experts": ref["experts"],
                                 "e_hq4": ref["e_hq4"], "hq4_off": off}
                    else:
                        rn_, kn_ = ref["shape"]
                        hq_meta = ref.get("hq")
                        if hq_meta:
                            length = rn_ * (kn_ // hq_meta["group"]) * any_block_bytes(
                                hq_meta["bits"], hq_meta["group"], is_shvq=(hq_meta.get("vq") == "shvq1"))
                        else:
                            length = rn_ * (kn_ // GROUP) * BLOCK_BYTES
                        entry = {"shape": ref["shape"], "hq4_off": off}
                    if "hq" in ref:
                        entry["hq"] = dict(ref["hq"])
                    rf.seek(ref["hq4_off"])
                    left = length
                    while left:
                        buf = rf.read(min(chunk, left))
                        if not buf:
                            raise ValueError(f"reference pack truncated at {name}")
                        f.write(buf)
                        left -= len(buf)
                    pad_q4g()
                    index[name] = entry

        hdr = json.dumps({"provenance": provenance, "tensors": index}).encode()
        f.seek(hdr_off)
        f.write(len(hdr).to_bytes(8, "little"))
        f.seek(0, 2)
        f.write(hdr)
    return out_path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_raw_plane(name: str) -> bool:
    return (any(t in name for t in (
                ".norm.", "_norm.", ".enorm.", ".hnorm.", "layernorm", "A_log", "dt_bias",
                "conv1d", "q_norm", "k_norm", "patch_embed", "embed_tokens.",
                ".hc_read.", ".hc_write.", ".hc_attn_", ".hc_ffn_", ".mhc_attn.", ".mhc_ffn.", "block_inject",
                ".mlp.gate.weight", "e_score_correction_bias", "shared_expert_gate.weight",
                "f_b_proj", "g_b_proj",
                "index_kpool_compress", "indexer.weights_proj"))
            or name.startswith("model.visual."))


def synthesize_base_chat_template(source: Path) -> str:
    tok_obj = json.loads((source / "tokenizer.json").read_text())
    pp = tok_obj.get("post_processor") or {}
    single = pp.get("single") if isinstance(pp, dict) and pp.get("type") == "TemplateProcessing" else None
    sp0 = single[0].get("SpecialToken", {}).get("id", "") if isinstance(single, list) and single else ""
    lead = json.dumps(sp0 if isinstance(sp0, str) else "")
    bo, bc, vo, vc = "{" + "%-", " %" + "}", "{" + "{-", " }" + "}"
    return "\n".join([
        f"{bo} set ns = namespace(multi=false) {bc}",
        f"{bo} for m in messages {bc}{bo} if m['role'] == 'assistant' {bc}{bo} set ns.multi = true {bc}{bo} endif {bc}{bo} endfor {bc}",
        f"{vo} {lead} {vc}",
        f"{bo} for m in messages {bc}",
        f"{bo} if not loop.first {bc}{vo} '\\n\\n' {vc}{bo} endif {bc}",
        f"{bo} if ns.multi and m['role'] == 'user' {bc}{vo} 'User: ' {vc}{bo} elif ns.multi and m['role'] == 'assistant' {bc}{vo} 'Assistant: ' {vc}{bo} endif {bc}",
        f"{bo} if m['content'] is string {bc}{vo} m['content'] {vc}{bo} elif m['content'] is iterable {bc}{bo} for it in m['content'] {bc}{bo} if 'text' in it {bc}{vo} it['text'] {vc}{bo} endif {bc}{bo} endfor {bc}{bo} endif {bc}",
        f"{bo} endfor {bc}",
        f"{bo} if ns.multi and add_generation_prompt {bc}{vo} '\\n\\nAssistant: ' {vc}{bo} endif {bc}",
    ])


def resolve_snapshot_dir(root: Path) -> Path:
    root = root.expanduser()
    snaps = root / "snapshots"
    if snaps.is_dir():
        ref_main = root / "refs" / "main"
        if ref_main.is_file():
            rev = ref_main.read_text().strip()
            cand = snaps / rev
            if cand.is_dir():
                return cand
        cands = sorted(
            [d for d in snaps.iterdir() if d.is_dir() and (d / "config.json").is_file()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0]
    return root


def quantize_model(source: Path, destination: Path,
                   reuse_ngram_from: Path | None = None,
                   hybrid_hq8_from: Path | None = None,
                   flat_root: bool = False,
                   ngram_quant: str = "bf16",
                   l1_hot: int = 0,
                   quant: str = "hq4_g32_m5",
                   group_size: int | None = None,
                   dense_quant: str | None = None,
                   expert_quant: str | None = None,
                   expert_down_quant: str | None = None):
    import torch as _t
    from safetensors import safe_open
    from safetensors.torch import load_file
    from verify import audit_pack, read_header

    dense_spec = parse_any_quant_spec(dense_quant or quant, default_group=group_size)
    expert_spec = parse_any_quant_spec(expert_quant or quant, default_group=group_size)
    expert_down_spec = parse_any_quant_spec(
        expert_down_quant or expert_quant or quant, default_group=group_size)
    dense_fmt = format_any_quant_spec(*dense_spec)
    expert_fmt = format_any_quant_spec(*expert_spec)
    expert_down_fmt = format_any_quant_spec(*expert_down_spec)
    if hybrid_hq8_from is not None:
        quant_recorded = "hq8_hq4_hybrid_v1"
    elif dense_fmt == expert_fmt == expert_down_fmt:
        quant_recorded = dense_fmt
    elif expert_fmt == expert_down_fmt:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}"
    else:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}__expdown_{expert_down_fmt}"

    _t.set_num_threads(min(os.cpu_count() or 8, 16))
    source = resolve_snapshot_dir(source).resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    if destination.suffix == ".qwp":
        raise ValueError("--out must be a model directory, not a .qwp file")
    if destination == source or source in destination.parents:
        raise ValueError("output must be outside the source checkpoint")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc0 = config["text_config"] if "text_config" in config else config
    if tc0.get("model_type") in ("ernie5_5_moe", "ernie5_5_moe_text"):
        from ernie55_adapter import quantize_ernie55_model
        return quantize_ernie55_model(
            source=source, destination=destination, flat_root=flat_root,
            quant=quant, group_size=group_size, dense_quant=dense_quant,
            expert_quant=expert_quant, expert_down_quant=expert_down_quant)
    qcfg = config.get("quantization_config")
    fp8_block_size = 128
    if qcfg is not None:
        if not isinstance(qcfg, dict) or qcfg.get("quant_method") != "fp8":
            raise ValueError("REFUSED: source is already quantized; use the original checkpoint")
        wbs = qcfg.get("weight_block_size")
        if isinstance(wbs, list) and len(wbs) == 2 and wbs[0] == wbs[1]:
            fp8_block_size = int(wbs[0])
    if not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    embedded_chat_template = None
    if not (source / "chat_template.jinja").is_file():
        tok_cfg_path = source / "tokenizer_config.json"
        if tok_cfg_path.is_file():
            tok_cfg = json.loads(tok_cfg_path.read_text())
            if isinstance(tok_cfg.get("chat_template"), str):
                embedded_chat_template = tok_cfg["chat_template"]
        if not embedded_chat_template:
            embedded_chat_template = synthesize_base_chat_template(source)

    shard_names, expected_shapes, raw_all_shapes = {}, {}, {}
    for path in files:
        with safe_open(path, framework="pt") as sf:
            shard_names[path] = list(sf.keys())
            for name in shard_names[path]:
                if name in raw_all_shapes:
                    raise ValueError(f"duplicate source tensor: {name}")
                raw_all_shapes[name] = list(sf.get_slice(name).get_shape())
                if not name.endswith("_scale_inv"):
                    expected_shapes[name] = raw_all_shapes[name]

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        actual_map = {name: path.name for path, names in shard_names.items() for name in names}
        if actual_map != weight_map:
            raise ValueError("source shards do not match model.safetensors.index.json")

    ple_shards, i64_constants = scan_ple_and_router(
        files, shard_names, expected_shapes, source)

    ref_tensors = None
    ref_qwp = None
    if hybrid_hq8_from is not None:
        ref_qwp = Path(hybrid_hq8_from) / "model.qwp"
        if not ref_qwp.is_file():
            raise FileNotFoundError(f"hybrid reference pack missing: {ref_qwp}")
        ref_tensors = read_header(ref_qwp)[0]["tensors"]
        for name, entry in ref_tensors.items():
            if "hq4_off" not in entry or "raw_off" in entry:
                continue
            if expected_shapes.get(name) != entry["shape"]:
                raise ValueError(
                    f"hybrid reference {name}: shape {entry['shape']} not in source")
        for name, shape in expected_shapes.items():
            if is_ple_ngram(name) or name in i64_constants:
                continue
            quant_candidate = (len(shape) == 2 and shape[1] % GROUP == 0
                               and not is_raw_plane(name))
            if quant_candidate and not hq8_target(name) and name not in ref_tensors:
                raise ValueError(
                    f"hybrid manifest: {name} is neither an HQ8 target nor an "
                    f"HQ4 plane in the reference pack (fail-closed)")

    sha256_source_hex = None
    if reuse_ngram_from is not None:
        reuse_qwp = Path(reuse_ngram_from) / "model.qwp"
        if reuse_qwp.is_file():
            reuse_prov = read_header(reuse_qwp)[0].get("provenance", {})
            if ((reuse_prov.get("source") == str(source)
                 or (reuse_prov.get("source_revision") and reuse_prov.get("source_revision") == source.name))
                    and reuse_prov.get("sha256_source")):
                sha256_source_hex = reuse_prov["sha256_source"]
                print(f"reusing sha256_source from {reuse_qwp}", flush=True)
    if sha256_source_hex is None and ref_qwp is not None:
        ref_prov = read_header(ref_qwp)[0].get("provenance", {})
        if (ref_prov.get("source") == str(source)
                and ref_prov.get("sha256_source")):
            sha256_source_hex = ref_prov["sha256_source"]
            print(f"reusing sha256_source from {ref_qwp}", flush=True)
    if sha256_source_hex is None:
        h = hashlib.sha256()
        for path in files:
            print(f"hashing {path.name}", flush=True)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(16 << 20), b""):
                    h.update(chunk)
        sha256_source_hex = h.hexdigest()

    tc = config["text_config"] if "text_config" in config else config
    if tc.get("model_type") in MODEL_ID and "num_hidden_layers" in tc:
        identity = {
            "model_id": MODEL_ID[tc["model_type"]],
            "geometry_digest": hashlib.sha256(geo_v2(tc).encode()).hexdigest(),
        }
    else:
        identity = {}
    stats = {"hq4_g32_m5": 0, "hq8_g32_m5": 0, "bf16": 0, "expert_hq4": 0,
             "ple_out_of_core": len(ple_shards)}

    def stream():
        for path in files:
            active_names = [
                n for n in shard_names[path]
                if not is_ple_ngram(n) and n not in i64_constants and not n.endswith("_scale_inv")
            ]
            if not active_names:
                continue
            print(f"quantizing {path.name}", flush=True)
            tensors = load_file(path)
            for name in active_names:
                arr = tensors.pop(name)
                sinv_key = f"{name}_scale_inv"
                if sinv_key in tensors:
                    arr = dequant_fp8_blockwise(arr, tensors.pop(sinv_key), block_size=fp8_block_size)
                if arr.ndim == 3 and not expert_stack(name, arr.shape):
                    if not (name.endswith("conv1d.weight")
                            or ".hc_read." in name
                            or ".hc_write." in name
                            or "block_inject" in name):
                        raise ValueError(
                            f"{name}: unexpected rank-3 shape {list(arr.shape)} — "
                            f"refuse to guess a layout")
                if expert_stack(name, arr.shape):
                    if hybrid_hq8_from is not None:
                        continue
                    stats["expert_hq4"] += 1
                    spec = expert_down_spec if "down_proj" in name else expert_spec
                    fmt = expert_down_fmt if "down_proj" in name else expert_fmt
                    stats[fmt] = stats.get(fmt, 0) + 1
                    yield name, arr, spec
                    continue
                if ".experts." in name and "shared_expert" not in name and arr.ndim == 2:
                    if hybrid_hq8_from is not None:
                        continue
                    stats["expert_hq4"] += 1
                    spec = expert_down_spec if "down_proj" in name else expert_spec
                    fmt = expert_down_fmt if "down_proj" in name else expert_fmt
                    stats[fmt] = stats.get(fmt, 0) + 1
                    yield name, arr, spec
                    continue
                raw = (is_raw_plane(name) or arr.ndim != 2 or arr.shape[1] % GROUP != 0)
                if hybrid_hq8_from is not None and not raw:
                    if hq8_target(name):
                        stats["hq8_g32_m5"] += 1
                        yield name, arr, "hq8"
                    continue
                if raw:
                    stats["bf16"] += 1
                    yield name, arr, True
                else:
                    stats[dense_fmt] = stats.get(dense_fmt, 0) + 1
                    yield name, arr, dense_spec
            del tensors
            try:
                import torch as _tmps
                if _tmps.backends.mps.is_available():
                    _tmps.mps.empty_cache()
            except Exception:
                pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-",
                                     dir=destination.parent) as temp:
        staging = Path(temp) / "model"
        # A flat root keeps the assets beside the weights; a legacy root
        # keeps them under `assets/`.
        assets = staging if flat_root else staging / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for src in sorted(source.iterdir()):
            if (not src.is_file() or src.name.startswith(".")
                    or src.name.endswith((".safetensors", ".index.json"))
                    or src.name == "crc32.txt"):
                continue
            shutil.copyfile(src, assets / src.name)
            manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
        if embedded_chat_template and not (assets / "chat_template.jinja").is_file():
            tpl_dst = assets / "chat_template.jinja"
            tpl_dst.write_text(embedded_chat_template, encoding="utf-8")
            manifest["chat_template.jinja"] = {"sha256": sha256(tpl_dst), "bytes": tpl_dst.stat().st_size}
        if ple_shards:
            loc_path = assets / PLE_LOCATION_NAME
            write_ple_location_file(ple_shards, loc_path,
                                    router_constants=i64_constants)
            manifest[PLE_LOCATION_NAME] = {
                "sha256": sha256(loc_path), "bytes": loc_path.stat().st_size
            }
            ngram_path = assets / NGRAM_BIN_NAME
            reused_ngram_sha = None
            ng_meta = None
            if reuse_ngram_from is not None:
                reuse_src = resolve_assets_dir(Path(reuse_ngram_from)) / NGRAM_BIN_NAME
                if not reuse_src.is_file():
                    raise FileNotFoundError(f"missing {reuse_src}")
                os.link(reuse_src, ngram_path)
                reuse_mf = resolve_assets_dir(Path(reuse_ngram_from)) / "manifest.json"
                if reuse_mf.is_file():
                    ng_meta = json.loads(reuse_mf.read_text()).get(NGRAM_BIN_NAME, {})
                    if (ng_meta.get("bytes") == ngram_path.stat().st_size
                            and ng_meta.get("sha256")):
                        reused_ngram_sha = ng_meta["sha256"]
            else:
                ng_meta = build_ngram_bin(ple_shards, source, ngram_path,
                                          quant=ngram_quant, l1_seed=l1_hot)
            manifest[NGRAM_BIN_NAME] = {
                "sha256": reused_ngram_sha or sha256(ngram_path),
                "bytes": ngram_path.stat().st_size,
            }
            if ng_meta:
                # Repeat header row format in manifest; older reused tables omit these keys.
                for key in ("version", "dtype", "row_bytes"):
                    if ng_meta.get(key) is not None:
                        manifest[NGRAM_BIN_NAME][key] = ng_meta[key]
        manifest_path = assets / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        weights = staging / "model.qwp"
        pack_file(stream(), weights, {
            **identity,
            "source": str(source), "sha256_source": sha256_source_hex,
            "source_revision": (source.name if source.parent.name == "snapshots"
                                and len(source.name) == 40
                                and all(c in "0123456789abcdef" for c in source.name)
                                else None),
            "quant_recorded": quant_recorded,
            "hq8_reference": (str(ref_qwp) if ref_qwp is not None else None),
            "assets_dir": "." if flat_root else "assets",
            "assets_manifest_sha256": sha256(manifest_path),
        }, copy_entries=(ref_qwp, ref_tensors) if hybrid_hq8_from is not None else None)
        audit_pack(weights, expected_shapes=expected_shapes)
        (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
        if destination.exists():
            raise FileExistsError(f"output appeared during quantization: {destination}")
        staging.rename(destination)
    print(f"packed quant={quant_recorded} stats={stats} -> {destination}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pos_model", nargs="?", default=None, help="optional positional source model directory")
    ap.add_argument("pos_out", nargs="?", default=None, help="optional positional output model directory")
    ap.add_argument("--selfpack", metavar="PATH")
    ap.add_argument("--selfpack-expert", metavar="PATH")
    ap.add_argument("--selfpack-hq8", metavar="PATH")
    ap.add_argument("--model", metavar="DIR")
    ap.add_argument("--out", metavar="DIR")
    ap.add_argument("--reuse-ngram-from", metavar="DIR", default=None)
    ap.add_argument("--hybrid-hq8-from", metavar="DIR", default=None,
                    help="cook a hybrid HQ8/HQ4 pack: HQ8 dense from master, "
                         "HQ4 planes copied byte-identically from this pack")
    ap.add_argument("--audit-ngram", metavar="PATH")
    ap.add_argument("--audit-release", metavar="DIR")
    ap.add_argument("--checkpoint", metavar="DIR")
    ap.add_argument("--post-pruning", action="store_true")
    ap.add_argument("--quant", default="hq4_g32_m5",
                    help="quantization format (e.g. hq4_g32_m5, hq4, hq6, hq8, hq6_g64)")
    ap.add_argument("--group-size", type=int, default=None, choices=list(VALID_GROUPS),
                    help="default quantization group size (32, 64, 128, 256)")
    ap.add_argument("--dense-quant", default=None,
                    help="override quantization format for 2D dense projections (e.g. hq8_g32)")
    ap.add_argument("--expert-quant", default=None,
                    help="override quantization format for MoE expert stacks (e.g. hq2_g128)")
    ap.add_argument("--expert-down-quant", default=None,
                    help="override quantization format for MoE expert down_proj (e.g. hq3_g128)")
    ap.add_argument("--ngram-quant", default="bf16", choices=["bf16", "q8", "q4", "vq2", "vq2_g32", "vq4", "vq4_g32"],
                    help="the QNGRAM row format of the PLE n-gram table: bf16 "
                         "(v1), q8 or q4 (v2), or vq2_g32/vq4_g32 (v3 e-means)")
    ap.add_argument("--l1-hot", type=int, default=0,
                    help="the QNGRAM header's l1Hot field: how many hot rows "
                         "the writer recorded")
    ap.add_argument("--flat-root", action="store_true",
                    help="write a FLAT model root: config.json, model.qwp and "
                         "ngram.bin side by side instead of under assets/")
    a = ap.parse_args()
    if a.audit_ngram:
        info = audit_ngram_bin(Path(a.audit_ngram))
        print(json.dumps(info, sort_keys=True))
        sys.exit(0)
    if a.audit_release:
        from audit_release import audit_release as _audit_release
        info = _audit_release(Path(a.audit_release),
                              checkpoint=Path(a.checkpoint) if a.checkpoint else None,
                              post_pruning=a.post_pruning)
        print(json.dumps(info, sort_keys=True))
        sys.exit(0)
    model_arg = a.model or a.pos_model
    out_arg = a.out or a.pos_out
    if model_arg:
        if not out_arg:
            ap.error("--model requires --out DIR")
        reuse = Path(a.reuse_ngram_from) if a.reuse_ngram_from else None
        hybrid = Path(a.hybrid_hq8_from) if a.hybrid_hq8_from else None
        quantize_model(Path(model_arg), Path(out_arg), reuse_ngram_from=reuse,
                       hybrid_hq8_from=hybrid, flat_root=a.flat_root,
                       ngram_quant=a.ngram_quant, l1_hot=a.l1_hot,
                       quant=a.quant, group_size=a.group_size,
                       dense_quant=a.dense_quant, expert_quant=a.expert_quant,
                       expert_down_quant=a.expert_down_quant)
        sys.exit(0)
    if a.selfpack:
        N, K = 256, 1024
        rr, cc = np.meshgrid(np.arange(N), np.arange(K), indexing="ij")
        w = (np.sin(rr * 0.013 + cc * 0.007) * 0.02).astype(np.float32)
        w[:, 5] *= np.float32(6.0)
        norm = np.sin(rr[:, 0] * 0.021) * 0.5 + 1.0
        pack_file({"w_test": w}, Path(a.selfpack),
                  {"source": "selfpack", "sha256_source": "n/a"},
                  raw={"norm_test": norm})
        blocks = quant_hq4_g32_m5(w)
        dq4 = dequant_hq4_g32_m5(blocks)
        np.save(a.selfpack + ".dq4.npy", dq4.astype(np.float32))
        import torch as _t
        np.save(a.selfpack + ".raw.npy",
                _t.from_numpy(np.ascontiguousarray(norm)).to(_t.bfloat16).float().numpy())
        print("selfpack:", a.selfpack)
        sys.exit(0)
    if a.selfpack_expert:
        import torch as _t
        E, N, K = 3, 4, 256
        ee, rr, cc = np.meshgrid(np.arange(E), np.arange(N), np.arange(K), indexing="ij")
        stack = (np.sin(ee * 0.11 + rr * 0.013 + cc * 0.007) * 0.02
                 + np.cos(ee * 0.37) * 0.01).astype(np.float32)
        stack[:, 1, 5] *= np.float32(6.0)
        down = (np.cos(ee * 0.23 - rr * 0.009 + cc * 0.011) * 0.03).astype(np.float32)
        norm = np.sin(ee[:, 0, 0] * 0.021 + 1.0) * 0.5 + 1.0
        name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        dname = "model.language_model.layers.0.mlp.experts.down_proj"
        sidecar = str(Path(a.selfpack_expert).with_suffix("")) + ".slices"
        pack_file({name: _t.from_numpy(stack), dname: _t.from_numpy(down)},
                  Path(a.selfpack_expert),
                  {"source": "selfpack-expert", "sha256_source": "n/a"},
                  raw={"ple_norm": _t.from_numpy(norm)},
                  sidecar_dir=sidecar)
        print("selfpack-expert:", a.selfpack_expert, "sidecars:", sidecar)
        sys.exit(0)
    if a.selfpack_hq8:
        N, K = 64, 256
        rr, cc = np.meshgrid(np.arange(N), np.arange(K), indexing="ij")
        w = (np.sin(rr * 0.017 + cc * 0.005) * 0.03).astype(np.float32)
        w[:, 7] *= np.float32(5.0)
        pack_file([("w_test8", w, "hq8")], Path(a.selfpack_hq8),
                  {"source": "selfpack-hq8", "sha256_source": "n/a"})
        qs, sc = quant_hq8_g32_m5(w)
        dequant_hq8_g32_m5_rot(qs, sc).astype("<f4").tofile(
            a.selfpack_hq8 + ".dq8rot.f32.bin")
        print("selfpack-hq8:", a.selfpack_hq8)
        sys.exit(0)
    rng = np.random.default_rng(7)
    N, K = 5120, 17408
    w = rng.normal(0, 0.02, (N, K)).astype(np.float32)
    w += (rng.standard_t(4, (N, K)) * 0.004).astype(np.float32)
    dq4 = dequant_hq4_g32_m5(quant_hq4_g32_m5(w))
    e4 = float(np.sqrt(((w - dq4) ** 2).mean()))
    w_rot = fwht32_ortho(w).reshape(-1, GROUP)
    rng_theory = ((2.0 * np.abs(w_rot).max(-1)) / 15.0) / math.sqrt(12)
    ok = e4 < 1.25 * rng_theory.mean()
    print("PACKER SANITY:", "PASS" if ok else "FAIL")

