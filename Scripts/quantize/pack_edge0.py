#!/usr/bin/env python3
"""Pack Edge0/Edge0-145B-NVFP4 into a Qwegn Group-Latent .qwp pack (< 90 GB)."""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from hq_blocks import _fwht32_torch
from pack import pack_file, sha256
from verify import audit_pack, read_header

NUM_LAYERS = 48
NUM_EXPERTS = 512
HIDDEN = 2560
LATENT_GROUPS = 8
LATENT_DIM = 1792
EXPERT_INTER = 640
Q8_MAT_BYTES = (LATENT_DIM * HIDDEN // 32) * 34
NVFP4_EXPERT_BYTES = 1_935_376
NVFP4_BLOCKS_BYTES = 1_935_360

_Q8_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
_E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
_E4M3_LUT = np.array(
    [
        (-1.0 if (u >> 7) else 1.0)
        * (((u & 7) / 8.0) * (2.0 ** -6) if ((u >> 3) & 0xF) == 0
           else (1.0 + (u & 7) / 8.0) * (2.0 ** (((u >> 3) & 0xF) - 7)))
        for u in range(256)
    ],
    dtype=np.float32,
)


def parse_native_experts(path: Path) -> list[int]:
    offsets = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        layer, off, rec_bytes = int(parts[0]), int(parts[3]), int(parts[4])
        if rec_bytes != NVFP4_EXPERT_BYTES:
            raise ValueError(f"layer {layer}: record size {rec_bytes} != {NVFP4_EXPERT_BYTES}")
        offsets[layer] = off
    if len(offsets) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers in {path}, got {len(offsets)}")
    return [offsets[l] for l in range(NUM_LAYERS)]


def dequant_q8_matrices(buf: bytes, count: int, rows: int, cols: int) -> np.ndarray:
    blocks = np.frombuffer(buf, dtype=_Q8_DTYPE, count=count * rows * cols // 32)
    out = blocks["d"].astype(np.float32)[:, None] * blocks["qs"].astype(np.float32)
    return np.ascontiguousarray(out.reshape(count, rows, cols))


def decode_nvfp4_layer(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(NUM_EXPERTS, NVFP4_EXPERT_BYTES)
    blocks = raw[:, :NVFP4_BLOCKS_BYTES].reshape(NUM_EXPERTS, 3, 17920, 36)
    scales_f32 = raw[:, NVFP4_BLOCKS_BYTES:].view(np.float32)[:, :3]
    sb = _E4M3_LUT[blocks[:, :, :, :4]] * scales_f32[:, :, None, None]
    qs = blocks[:, :, :, 4:].reshape(NUM_EXPERTS, 3, 17920, 4, 8)
    w = np.empty((NUM_EXPERTS, 3, 17920, 4, 16), dtype=np.float32)
    w[..., :8] = _E2M1[qs & 0xF]
    w[..., 8:] = _E2M1[qs >> 4]
    w *= sb[..., None]
    gate_up = np.ascontiguousarray(w[:, :2].reshape(NUM_EXPERTS, 2 * EXPERT_INTER, LATENT_DIM))
    down = np.ascontiguousarray(w[:, 2].reshape(NUM_EXPERTS, LATENT_DIM, EXPERT_INTER))
    return gate_up, down


def fold_output_wht(arr: np.ndarray, chunk: int = 64) -> torch.Tensor:
    e_cnt, out_dim, in_dim = arr.shape
    assert out_dim == LATENT_DIM
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    out = torch.empty((e_cnt, out_dim, in_dim), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, e_cnt, chunk):
            j = min(i + chunk, e_cnt)
            t = torch.from_numpy(arr[i:j]).to(device=dev, dtype=torch.float32)
            t = _fwht32_torch(t.reshape(j - i, LATENT_DIM // 32, 32, in_dim).transpose(2, 3))
            out[i:j] = t.transpose(2, 3).reshape(j - i, LATENT_DIM, in_dim).cpu()
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_pack(src_dir: Path, ref_dir: Path, out_dir: Path, max_bytes: int) -> None:
    ref_qwp = ref_dir / "model.qwp"
    ref_hdr, _ = read_header(ref_qwp)
    ref_tensors = ref_hdr["tensors"]
    bad_hq8 = [k for k, v in ref_tensors.items() if "hq8_off" in v]
    if bad_hq8:
        raise ValueError(f"unexpected legacy hq8_off tensors in ref: {bad_hq8[:3]}")

    copy_ref = {
        k: v for k, v in ref_tensors.items()
        if not (k.startswith("model.language_model.layers.") and ".mlp.experts." in k)
    }
    labels = np.fromfile(src_dir / "labels.bin", dtype=np.int32).reshape(NUM_LAYERS, NUM_EXPERTS)
    layer_offsets = parse_native_experts(src_dir / "native_experts.txt")

    def iter_tensors():
        with ref_qwp.open("rb") as rf:
            for name in sorted(copy_ref):
                entry = copy_ref[name]
                if "raw_off" not in entry:
                    continue
                rf.seek(entry["raw_off"])
                n_elems = math.prod(entry["shape"])
                u16 = np.frombuffer(rf.read(n_elems * 2), dtype=np.uint16).copy()
                t = torch.from_numpy(u16).view(torch.bfloat16).reshape(entry["shape"])
                yield name, t, True

        t0 = time.monotonic()
        with (src_dir / "latent_dense_q8.bin").open("rb") as lf, (src_dir / "experts.bin").open("rb") as ef:
            for l in range(NUM_LAYERS):
                lbl_t = torch.from_numpy(labels[l].astype(np.float32))
                yield f"model.language_model.layers.{l}.mlp.latent_labels", lbl_t, True

                lf.seek(l * 2 * LATENT_GROUPS * Q8_MAT_BYTES)
                d_mat = dequant_q8_matrices(lf.read(LATENT_GROUPS * Q8_MAT_BYTES), LATENT_GROUPS, LATENT_DIM, HIDDEN)
                u_mat = dequant_q8_matrices(lf.read(LATENT_GROUPS * Q8_MAT_BYTES), LATENT_GROUPS, HIDDEN, LATENT_DIM)
                yield f"model.language_model.layers.{l}.mlp.experts.latent_d", fold_output_wht(d_mat), "hq8_g32"
                yield f"model.language_model.layers.{l}.mlp.experts.latent_u", torch.from_numpy(u_mat), "hq8_g32"

                ef.seek(layer_offsets[l])
                gate_up, down = decode_nvfp4_layer(ef.read(NUM_EXPERTS * NVFP4_EXPERT_BYTES))
                yield f"model.language_model.layers.{l}.mlp.experts.gate_up_proj", torch.from_numpy(gate_up), "hq4_g64"
                yield f"model.language_model.layers.{l}.mlp.experts.down_proj", fold_output_wht(down), "hq4_g64"
                print(f"layer {l + 1:2d}/{NUM_LAYERS} packed ({time.monotonic() - t0:.1f}s)", flush=True)

    ref_prov = dict(ref_hdr["provenance"])
    provenance = {
        "model_id": ref_prov["model_id"],
        "geometry_digest": ref_prov["geometry_digest"],
        "source": "hf://Edge0/Edge0-145B-NVFP4",
        "sha256_source": hashlib.sha256(b"Edge0/Edge0-145B-NVFP4").hexdigest(),
        "quant_recorded": "edge0_latent_hq8_g32__expert_hq4_g64",
        "hq8_reference": str(ref_qwp),
        "assets_dir": "assets",
        "assets_manifest_sha256": ref_prov["assets_manifest_sha256"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_qwp = out_dir / "model.qwp"
    pack_file(iter_tensors(), out_qwp, provenance=provenance, copy_entries=(ref_qwp, copy_ref))

    for fname in ("model.qlsa", "LICENSE", "README.md"):
        src_f = ref_dir / fname
        if src_f.is_file():
            link_or_copy(src_f, out_dir / fname)
    for asset_f in sorted((ref_dir / "assets").iterdir()):
        if asset_f.is_file():
            link_or_copy(asset_f, out_dir / "assets" / asset_f.name)

    digest = sha256(out_qwp)
    (out_dir / "model.qwp.sha256").write_text(f"{digest}  model.qwp\n")
    _, counts = audit_pack(out_qwp)
    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"Verified {out_qwp}: counts={counts}")
    print(f"Total pack footprint: {total_bytes:,} B ({total_bytes / 1e9:.3f} GB)")
    if total_bytes >= max_bytes:
        raise ValueError(f"pack size {total_bytes} B exceeds budget {max_bytes} B")


def main() -> None:
    p = argparse.ArgumentParser(description="Pack Edge0-145B-NVFP4 into .qwp (<90 GB)")
    p.add_argument("--src", type=Path, default=Path.home() / ".qwg/cache/Edge0-145B-NVFP4")
    p.add_argument("--ref", type=Path, default=Path.home() / ".qwg/models/Qwen3.8-Flash-Next-hq8g32-hq4g64")
    p.add_argument("--out", type=Path, default=Path.home() / ".qwg/models/Edge0-145B-hq8g32-hq4g64")
    p.add_argument("--max-bytes", type=int, default=90_000_000_000)
    args = p.parse_args()
    build_pack(args.src, args.ref, args.out, args.max_bytes)


if __name__ == "__main__":
    main()
