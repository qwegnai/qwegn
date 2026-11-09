#!/usr/bin/env python3
"""Offline Latent Sparse Attention (.qlsa) Sidecar Converter for qwegine.

Builds a geometry-verified `QLSA0001` (`model.qlsa`) binary sidecar containing
per-full-attention-layer low-rank index projections (`layers.<l>.wq_idx` and
`layers.<l>.wk_idx`, shape `[index_dim, lsa_input_dim]`, dtype `f16`) from a
packed `.qwp` (`QWP1`) or HuggingFace `.safetensors` checkpoint.

Supported conversion methods (`--method`):
  - `svd`:     Training-free SVD of the effective unrotated key projection
               `diag(1 + k_norm) @ k_proj[unrot]` (instant, zero calibration data).
  - `pca`:     Calibration-activated PCA of unrotated post-k_norm key states
               (uses `--calibration-file` or synthetic unit-sphere activations).
  - `distill`: Lightweight contrastive/KL InfoNCE alignment of `(wq_idx, wk_idx)`
               initialized from SVD so 4-token pooled dot products track max-pooled
               teacher attention logits.
  - `import`:  Imports pre-trained `layers.<l>.wq_idx` / `layers.<l>.wk_idx` tensors
               from a `.safetensors` or `.pt` file (`--import-path`).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

try:
    import numpy as np
except ModuleNotFoundError:
    if __name__ == "__main__":
        import os
        import shutil
        venv_py = REPO_ROOT / ".venv-oracle" / "bin" / "python"
        if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
            os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
        uv_bin = shutil.which("uv")
        if uv_bin and os.environ.get("QWG_UV_REEXEC") != "1":
            os.environ["QWG_UV_REEXEC"] = "1"
            os.execv(
                uv_bin,
                [uv_bin, "run", "--with", "numpy", "--with", "torch", "python3", str(Path(__file__).resolve()), *sys.argv[1:]],
            )
    raise

sys.path.insert(0, str(SCRIPT_DIR / "quantize"))
sys.path.insert(0, str(SCRIPT_DIR / "oracle"))

from hq_blocks import VALID_BITS, VALID_GROUPS, dequant_hq_blocks, hq_block_bytes  # noqa: E402
from hq4 import dequant_hq4_g32_m5  # noqa: E402

QLSA_MAGIC = b"QLSA"
QLSA_VERSION = 1
QLSA_ALIGNMENT = 64


def _f64hex(x: float) -> str:
    return "0x" + format(struct.unpack(">Q", struct.pack(">d", float(x)))[0], "x")


def _f32hex(x: float) -> str:
    return "0x" + format(struct.unpack(">I", struct.pack(">f", float(x)))[0], "x")


def compute_geometry_digest(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compute the canonical `qwg-geo-v2` SHA-256 hex digest matching QWModelDescriptor."""
    is_hybrid = (
        "text_config" in cfg
        or "full_attention_interval" in cfg
        or "linear_key_head_dim" in cfg
        or "indexer_n_heads" in cfg
    )
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    model_type = tc["model_type"]
    num_layers = int(tc["num_hidden_layers"])
    raw_layer_types = tc.get("layer_types") or (["full_attention"] * num_layers)
    layer_kinds = "".join("F" if t == "full_attention" else "L" for t in raw_layer_types)
    fa_interval = int(tc.get("full_attention_interval", 1))
    q_heads = int(tc["num_attention_heads"])
    kv_heads = int(tc.get("num_key_value_heads", q_heads))
    hd = int(tc.get("head_dim") or (int(tc["hidden_size"]) // q_heads))
    rot_f = float(tc.get("partial_rotary_factor", 1.0))
    rope_params = tc.get("rope_parameters") or {}
    default_theta = 1e7 if is_hybrid else 10000.0
    theta = float(
        tc.get("rope_theta")
        or rope_params.get("rope_theta")
        or default_theta
    )
    if tc.get("attn_output_gate") is not None:
        gate_on = bool(tc["attn_output_gate"])
    else:
        gate_on = (
            tc.get("output_gate_type") is not None
            or model_type in (
                "qwen3_5",
                "qwen3_5_text",
                "qwen3_5_moe",
                "qwen3_5_moe_text",
                "sensenova_loop",
                "sensenova_loop_text",
                "qwen4_exp",
                "qwen4_exp_text",
            )
        )
    qgate = 1 if gate_on else 0
    gdn_qk = int(tc.get("linear_num_key_heads") or 0)
    gdn_v = int(tc.get("linear_num_value_heads") or 0)
    gdn_hd = int(tc.get("linear_key_head_dim") or 0)
    gdn_ck = int(tc.get("linear_conv_kernel_dim") or 0)
    mtp_layers = max(
        0,
        int(tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers") or 0),
    )
    mtp_ded = 1 if tc.get("mtp_use_dedicated_embeddings") else 0
    eos_raw = tc.get("eos_token_id", cfg.get("eos_token_id", 0))
    eos = int(eos_raw[0] if isinstance(eos_raw, list) else eos_raw)
    rms_eps = float(tc.get("rms_norm_eps", 1e-6))
    max_pos = int(tc.get("max_position_embeddings", 32768))

    z = lambda k: tc.get(k) if tc.get(k) is not None else 0
    eg = "sigmoid" if model_type in ("qwen4_exp", "qwen4_exp_text") else "-"
    ple_ids = tc.get("ple_layer_ids") or []

    fields = [
        f"model_type={model_type}",
        f"hidden={int(tc['hidden_size'])}",
        f"vocab={int(tc['vocab_size'])}",
        f"layers={num_layers}",
        f"layer_kinds={layer_kinds}",
        f"fa_interval={fa_interval}",
        f"q={q_heads}",
        f"kv={kv_heads}",
        f"hd={hd}",
        f"qgate={qgate}",
        f"rot_f={_f64hex(rot_f)}",
        f"theta={_f64hex(theta)}",
        f"gdn={gdn_qk},{gdn_v},{gdn_hd},{gdn_ck}",
        f"mtp={mtp_layers},{mtp_ded}",
        f"eos={eos}",
        f"rms={_f32hex(rms_eps)}",
        f"maxpos={max_pos}",
        f"idx={z('indexer_n_heads')},{z('indexer_kv_heads')},{z('indexer_head_dim')},{z('indexer_budget')},{z('indexer_compress_ratio')}",
        f"moe={z('num_experts')},{z('num_experts_per_tok')},{z('moe_intermediate_size')},{z('shared_expert_intermediate_size')},{eg}",
        f"hc={z('hc_count')},{z('hc_lowrank')}",
        "ple={},{},{},{},{},{},{},{}".format(
            ",".join(str(i) for i in ple_ids),
            z("ple_embed_dim"),
            z("ple_conv_kernel_size"),
            z("ngram_size"),
            z("heads_per_ngram"),
            z("ngram_vocab_size_base"),
            z("make_ngram_vocab_size_divisible_by"),
            z("split_ngram_parts"),
        ),
    ]
    canonical = "qwg-geo-v2:" + "|".join(fields)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    rotary_dims = int(float(hd) * rot_f)
    unrot = hd - rotary_dims
    slice_start = rotary_dims if unrot >= 64 else 0
    slice_dim = hd - slice_start
    full_attn_layers = [
        i for i, kind in enumerate(raw_layer_types) if kind == "full_attention"
    ]

    meta = {
        "model_type": model_type,
        "hidden_size": int(tc["hidden_size"]),
        "num_layers": num_layers,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": hd,
        "rotary_dims": rotary_dims,
        "lsa_slice_start": slice_start,
        "lsa_slice_dim": slice_dim,
        "lsa_input_dim": kv_heads * slice_dim,
        "full_attention_layers": full_attn_layers,
        "has_qk_norm": True,
    }
    return digest, meta


class WeightReader:
    """Reads and dequantizes 1D/2D tensors from either `model.qwp` or `.safetensors`."""

    def __init__(self, model_dir: Path, qwp_path: Path | None = None):
        self.model_dir = model_dir
        self.qwp_path = qwp_path
        self.qwp_header: dict[str, Any] | None = None
        self.qwp_tensors: dict[str, Any] = {}
        self.safetensor_map: dict[str, Path] = {}

        if qwp_path is not None and qwp_path.is_file():
            with qwp_path.open("rb") as f:
                if f.read(4) != b"QWP1":
                    raise ValueError(f"Invalid QWP1 magic in {qwp_path}")
                hdr_size = int.from_bytes(f.read(8), "little")
                file_len = f.seek(0, 2)
                f.seek(file_len - hdr_size)
                self.qwp_header = json.loads(f.read(hdr_size))
                self.qwp_tensors = self.qwp_header.get("tensors", {})
        else:
            idx_path = model_dir / "model.safetensors.index.json"
            if idx_path.is_file():
                idx = json.loads(idx_path.read_text())
                for k, fname in idx.get("weight_map", {}).items():
                    self.safetensor_map[k] = model_dir / fname
            else:
                for st_file in sorted(model_dir.glob("*.safetensors")):
                    self.safetensor_map["*"] = st_file

    def _find_key(self, candidates: list[str]) -> str:
        pool = self.qwp_tensors if self.qwp_header is not None else self.safetensor_map
        for c in candidates:
            if c in pool:
                return c
        if "*" in self.safetensor_map:
            return candidates[0]
        raise KeyError(f"None of {candidates} found in checkpoint")

    def read_tensor_f32(self, candidates: list[str]) -> np.ndarray:
        if self.qwp_header is not None:
            key = self._find_key(candidates)
            entry = self.qwp_tensors[key]
            shape = tuple(entry["shape"])
            assert self.qwp_path is not None
            with self.qwp_path.open("rb") as f:
                if "raw_off" in entry:
                    f.seek(int(entry["raw_off"]))
                    n_elem = math.prod(shape)
                    raw_u16 = np.frombuffer(f.read(n_elem * 2), dtype=np.uint16)
                    if entry.get("kind") == "f16":
                        return raw_u16.view(np.float16).astype(np.float32).reshape(shape)
                    u32 = raw_u16.astype(np.uint32) << 16
                    return u32.view(np.float32).reshape(shape)
                if "hq4_off" in entry:
                    n, k = shape
                    hq = entry.get("hq")
                    if hq is None:
                        bbytes = 20
                        grp = 32
                        f.seek(int(entry["hq4_off"]))
                        nb = k // grp
                        raw = np.frombuffer(f.read(n * nb * bbytes), dtype=np.uint8).reshape(
                            n, nb, bbytes
                        )
                        return dequant_hq4_g32_m5(raw)
                    bits = int(hq["bits"])
                    grp = int(hq["group"])
                    bbytes = hq_block_bytes(bits, grp)
                    nb = k // grp
                    f.seek(int(entry["hq4_off"]))
                    raw = np.frombuffer(f.read(n * nb * bbytes), dtype=np.uint8).reshape(
                        n, nb, bbytes
                    )
                    return dequant_hq_blocks(raw, bits=bits, group=grp, is_m5=False)
                raise ValueError(f"Unsupported QWP tensor plane for {key}: {entry}")

        from safetensors import safe_open

        for cand in candidates:
            st_path = self.safetensor_map.get(cand) or self.safetensor_map.get("*")
            if st_path is None:
                continue
            with safe_open(str(st_path), framework="pt", device="cpu") as sf:
                if cand in sf.keys():
                    return sf.get_tensor(cand).float().numpy()
        raise KeyError(f"Tensor {candidates} not found in safetensors")


def _layer_candidates(layer: int, suffix: str) -> list[str]:
    return [
        f"model.language_model.layers.{layer}.{suffix}",
        f"model.layers.{layer}.{suffix}",
        f"language_model.model.layers.{layer}.{suffix}",
        f"layers.{layer}.{suffix}",
    ]


def compute_top_singular_vectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Compute top-`rank` left singular vectors of `matrix` `[rows, cols]`, returning `[rank, rows]`."""
    u, _, _ = np.linalg.svd(matrix.astype(np.float32), full_matrices=False)
    actual = min(rank, u.shape[1])
    out = np.zeros((rank, matrix.shape[0]), dtype=np.float32)
    out[:actual, :] = u[:, :actual].T
    return out


def build_layer_svd(
    wk: np.ndarray,
    k_norm: np.ndarray | None,
    meta: dict[str, Any],
    index_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute data-free SVD basis `[index_dim, lsa_input_dim]` for a single full-attention layer."""
    kv_heads = meta["kv_heads"]
    hd = meta["head_dim"]
    s_start = meta["lsa_slice_start"]
    s_dim = meta["lsa_slice_dim"]
    hidden = meta["hidden_size"]

    wk_3d = wk.reshape(kv_heads, hd, hidden)
    if k_norm is not None:
        scale = (1.0 + k_norm.reshape(-1))[s_start : s_start + s_dim]
    else:
        scale = np.ones((s_dim,), dtype=np.float32)

    if kv_heads * s_dim <= 512:
        # Joint across all KV heads
        sliced = wk_3d[:, s_start : s_start + s_dim, :] * scale[None, :, None]
        joint = sliced.reshape(kv_heads * s_dim, hidden)
        basis = compute_top_singular_vectors(joint, index_dim)
    else:
        # Per-KV-head SVD averaged and re-orthogonalized across heads
        u_accum = np.zeros((index_dim, s_dim), dtype=np.float32)
        for h in range(kv_heads):
            sub = wk_3d[h, s_start : s_start + s_dim, :] * scale[:, None]
            u_h = compute_top_singular_vectors(sub, index_dim)
            if h > 0:
                dots = np.sum(u_h * u_accum, axis=1, keepdims=True)
                u_h = np.where(dots < 0.0, -u_h, u_h)
            u_accum += u_h
        q_orth, _ = np.linalg.qr(u_accum.T, mode="reduced")
        u_orth = q_orth.T
        basis = np.zeros((index_dim, kv_heads * s_dim), dtype=np.float32)
        inv_sqrt_kv = 1.0 / math.sqrt(float(kv_heads))
        actual = min(index_dim, u_orth.shape[0])
        for h in range(kv_heads):
            basis[:actual, h * s_dim : (h + 1) * s_dim] = u_orth[:actual, :] * inv_sqrt_kv

    f16 = basis.astype(np.float16)
    return f16, f16.copy()


def build_layer_pca(
    wk: np.ndarray,
    k_norm: np.ndarray | None,
    calib_h: np.ndarray,
    meta: dict[str, Any],
    index_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration-PCA basis `[index_dim, lsa_input_dim]` from calibration hidden states `calib_h`."""
    kv_heads = meta["kv_heads"]
    hd = meta["head_dim"]
    s_start = meta["lsa_slice_start"]
    s_dim = meta["lsa_slice_dim"]

    # Project calibration states [N, hidden] -> [N, kv_heads, hd]
    k_act = (calib_h @ wk.T).reshape(-1, kv_heads, hd)
    rms = np.sqrt(np.mean(k_act * k_act, axis=-1, keepdims=True) + 1e-6)
    k_normed = k_act / rms
    if k_norm is not None:
        k_normed = k_normed * (1.0 + k_norm.reshape(1, 1, hd))
    k_slice = k_normed[:, :, s_start : s_start + s_dim]
    k_slice = k_slice - np.mean(k_slice, axis=0, keepdims=True)

    u_accum = np.zeros((index_dim, s_dim), dtype=np.float32)
    for h in range(kv_heads):
        # Covariance [s_dim, s_dim]
        cov = (k_slice[:, h, :].T @ k_slice[:, h, :]) / max(1, k_slice.shape[0])
        u_h = compute_top_singular_vectors(cov, index_dim)
        if h > 0:
            dots = np.sum(u_h * u_accum, axis=1, keepdims=True)
            u_h = np.where(dots < 0.0, -u_h, u_h)
        u_accum += u_h

    q_orth, _ = np.linalg.qr(u_accum.T, mode="reduced")
    u_orth = q_orth.T
    basis = np.zeros((index_dim, kv_heads * s_dim), dtype=np.float32)
    inv_sqrt_kv = 1.0 / math.sqrt(float(kv_heads))
    actual = min(index_dim, u_orth.shape[0])
    for h in range(kv_heads):
        basis[:actual, h * s_dim : (h + 1) * s_dim] = u_orth[:actual, :] * inv_sqrt_kv

    f16 = basis.astype(np.float16)
    return f16, f16.copy()


def build_layer_distill(
    wq: np.ndarray,
    wk: np.ndarray,
    q_norm: np.ndarray | None,
    k_norm: np.ndarray | None,
    calib_h: np.ndarray,
    meta: dict[str, Any],
    index_dim: int,
    steps: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Contrastive / KL distillation starting from SVD basis using PyTorch."""
    import torch
    import torch.nn.functional as F

    wq_init, wk_init = build_layer_svd(wk, k_norm, meta, index_dim)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    kv_heads = meta["kv_heads"]
    q_heads = meta["q_heads"]
    gqa = q_heads // kv_heads
    hd = meta["head_dim"]
    s_start = meta["lsa_slice_start"]
    s_dim = meta["lsa_slice_dim"]

    h_t = torch.from_numpy(calib_h).to(device=device, dtype=torch.float32)
    n_tok = (h_t.shape[0] // 4) * 4
    h_t = h_t[:n_tok]

    wk_t = torch.from_numpy(wk).to(device=device, dtype=torch.float32)
    wq_t = torch.from_numpy(wq[: q_heads * hd]).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        q_raw = (h_t @ wq_t.T).reshape(n_tok, q_heads, hd)
        k_raw = (h_t @ wk_t.T).reshape(n_tok, kv_heads, hd)
        q_rms = torch.rsqrt(torch.mean(q_raw * q_raw, dim=-1, keepdim=True) + 1e-6)
        k_rms = torch.rsqrt(torch.mean(k_raw * k_raw, dim=-1, keepdim=True) + 1e-6)
        q_n = q_raw * q_rms
        k_n = k_raw * k_rms
        if q_norm is not None:
            q_n = q_n * (1.0 + torch.from_numpy(q_norm).to(device=device).reshape(1, 1, hd))
        if k_norm is not None:
            k_n = k_n * (1.0 + torch.from_numpy(k_norm).to(device=device).reshape(1, 1, hd))

        # Teacher attention logits [n_tok, n_blocks] via GQA mean + 4-token max pool
        q_gqa = q_n.reshape(n_tok, kv_heads, gqa, hd)
        teacher_logits = torch.einsum("tkgh,skh->tskg", q_gqa, k_n) / math.sqrt(float(hd))
        teacher_mean = teacher_logits.mean(dim=(-1, -2))  # [n_tok, n_tok]
        teacher_blocks = teacher_mean.reshape(n_tok, n_tok // 4, 4).amax(dim=-1)
        teacher_probs = F.softmax(teacher_blocks, dim=-1)

        # Unrotated slice inputs for student
        q_slice = q_gqa[:, :, :, s_start : s_start + s_dim].mean(dim=2).reshape(n_tok, kv_heads * s_dim)
        k_slice = k_n[:, :, s_start : s_start + s_dim].reshape(n_tok, kv_heads * s_dim)

    wq_param = torch.nn.Parameter(torch.from_numpy(wq_init.astype(np.float32)).to(device))
    wk_param = torch.nn.Parameter(torch.from_numpy(wk_init.astype(np.float32)).to(device))
    opt = torch.optim.AdamW([wq_param, wk_param], lr=3e-3, weight_decay=1e-4)
    temp = torch.nn.Parameter(torch.tensor(4.0, device=device))

    for _ in range(steps):
        opt.zero_grad()
        q_idx = F.normalize(q_slice @ wq_param.T, p=2, dim=-1)
        k_idx = (k_slice @ wk_param.T).reshape(n_tok // 4, 4, index_dim).mean(dim=1)
        k_idx = F.normalize(k_idx, p=2, dim=-1)
        student_logits = (q_idx @ k_idx.T) * temp
        log_p = F.log_softmax(student_logits, dim=-1)
        loss = F.kl_div(log_p, teacher_probs, reduction="batchmean")
        loss.backward()
        opt.step()

    return (
        wq_param.detach().cpu().numpy().astype(np.float16),
        wk_param.detach().cpu().numpy().astype(np.float16),
    )


def write_qlsa_sidecar(
    out_path: Path,
    provenance: dict[str, Any],
    tensors: dict[str, np.ndarray],
) -> None:
    """Write a `QLSA0001` binary sidecar atomically."""
    sorted_names = sorted(tensors.keys())
    cursor = 16  # 4B magic + 4B version + 8B header_size
    tensor_entries: dict[str, Any] = {}
    payload_chunks: list[tuple[int, bytes]] = []

    for name in sorted_names:
        arr = np.ascontiguousarray(tensors[name], dtype=np.float16)
        rem = cursor % QLSA_ALIGNMENT
        if rem != 0:
            cursor += QLSA_ALIGNMENT - rem
        raw_bytes = arr.tobytes()
        tensor_entries[name] = {
            "shape": list(arr.shape),
            "dtype": "f16",
            "raw_off": cursor,
            "bytes": len(raw_bytes),
        }
        payload_chunks.append((cursor, raw_bytes))
        cursor += len(raw_bytes)

    rem = cursor % QLSA_ALIGNMENT
    if rem != 0:
        cursor += QLSA_ALIGNMENT - rem

    header_obj = {
        "provenance": provenance,
        "tensors": tensor_entries,
    }
    header_bytes = json.dumps(header_obj, indent=2, sort_keys=True).encode("utf-8")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(QLSA_MAGIC)
        f.write(struct.pack("<I", QLSA_VERSION))
        f.write(struct.pack("<Q", len(header_bytes)))
        pos = 16
        for off, data in payload_chunks:
            if off > pos:
                f.write(b"\x00" * (off - pos))
            f.write(data)
            pos = off + len(data)
        if cursor > pos:
            f.write(b"\x00" * (cursor - pos))
        f.write(header_bytes)

    tmp_path.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a .qlsa Latent Sparse Attention sidecar.")
    parser.add_argument("--model", required=True, help="Model directory or path to model.qwp")
    parser.add_argument(
        "--method",
        choices=("svd", "pca", "distill", "import"),
        default="svd",
        help="Projection synthesis method (default: svd)",
    )
    parser.add_argument("--out", default=None, help="Output .qlsa path (default: <model_dir>/model.qlsa)")
    parser.add_argument("--index-dim", type=int, default=64, help="LSA index dimension (default: 64)")
    parser.add_argument(
        "--recommended-budget",
        type=int,
        default=2048,
        help="Recommended LSA token budget stored in provenance (default: 2048)",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Optional .npy calibration hidden states [N, hidden_size] for pca/distill",
    )
    parser.add_argument(
        "--import-path",
        default=None,
        help="Path to .safetensors file with pre-trained layers.<l>.wq_idx / wk_idx (--method import)",
    )
    args = parser.parse_args()

    model_arg = Path(args.model).expanduser().resolve()
    if model_arg.is_file():
        model_dir = model_arg.parent
        qwp_path: Path | None = model_arg
    else:
        model_dir = model_arg
        cand_qwp = model_dir / "model.qwp"
        qwp_path = cand_qwp if cand_qwp.is_file() else None

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        cfg_path = model_dir / "assets" / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in {model_dir} or {model_dir / 'assets'}")

    cfg = json.loads(cfg_path.read_text())
    geo_digest, meta = compute_geometry_digest(cfg)
    reader = WeightReader(model_dir, qwp_path=qwp_path)

    # Prefer geometry_digest embedded in QWP1 provenance when available and non-empty
    if reader.qwp_header is not None:
        qwp_prov = reader.qwp_header.get("provenance") or {}
        if isinstance(qwp_prov.get("geometry_digest"), str) and len(qwp_prov["geometry_digest"]) == 64:
            geo_digest = qwp_prov["geometry_digest"]

    out_path = Path(args.out).expanduser().resolve() if args.out else (model_dir / "model.qlsa")
    full_layers = meta["full_attention_layers"]
    index_dim = int(args.index_dim)
    lsa_input_dim = int(meta["lsa_input_dim"])

    print(
        f"[convert_lsa_sidecar] model={model_dir.name} method={args.method} "
        f"full_attn_layers={len(full_layers)} index_dim={index_dim} lsa_input_dim={lsa_input_dim}"
    )

    calib_h: np.ndarray | None = None
    if args.method in ("pca", "distill"):
        if args.calibration_file:
            calib_h = np.load(args.calibration_file).astype(np.float32)
        else:
            rng = np.random.default_rng(42)
            calib_h = rng.standard_normal((512, meta["hidden_size"]), dtype=np.float32) / math.sqrt(
                float(meta["hidden_size"])
            )

    tensors: dict[str, np.ndarray] = {}
    if args.method == "import":
        if not args.import_path:
            raise ValueError("--import-path is required when --method import is used")
        from safetensors import safe_open

        with safe_open(str(Path(args.import_path).expanduser().resolve()), framework="np") as sf:
            for l in full_layers:
                wq_key = f"layers.{l}.wq_idx"
                wk_key = f"layers.{l}.wk_idx"
                wq_arr = sf.get_tensor(wq_key).astype(np.float16)
                wk_arr = sf.get_tensor(wk_key).astype(np.float16)
                if wq_arr.shape != (index_dim, lsa_input_dim) or wk_arr.shape != (index_dim, lsa_input_dim):
                    raise ValueError(
                        f"Layer {l} imported shape {wq_arr.shape} != expected {(index_dim, lsa_input_dim)}"
                    )
                tensors[wq_key] = wq_arr
                tensors[wk_key] = wk_arr
    else:
        for l in full_layers:
            wk = reader.read_tensor_f32(_layer_candidates(l, "self_attn.k_proj.weight"))
            try:
                k_norm = reader.read_tensor_f32(_layer_candidates(l, "self_attn.k_norm.weight"))
            except KeyError:
                k_norm = None

            if args.method == "svd":
                wq_idx, wk_idx = build_layer_svd(wk, k_norm, meta, index_dim)
            elif args.method == "pca":
                assert calib_h is not None
                wq_idx, wk_idx = build_layer_pca(wk, k_norm, calib_h, meta, index_dim)
            elif args.method == "distill":
                assert calib_h is not None
                wq = reader.read_tensor_f32(_layer_candidates(l, "self_attn.q_proj.weight"))
                try:
                    q_norm = reader.read_tensor_f32(_layer_candidates(l, "self_attn.q_norm.weight"))
                except KeyError:
                    q_norm = None
                wq_idx, wk_idx = build_layer_distill(wq, wk, q_norm, k_norm, calib_h, meta, index_dim)
            else:
                raise ValueError(f"Unknown method: {args.method}")

            tensors[f"layers.{l}.wq_idx"] = wq_idx
            tensors[f"layers.{l}.wk_idx"] = wk_idx

    provenance = {
        "format_version": 1,
        "model_id": model_dir.name,
        "geometry_digest": geo_digest,
        "method": args.method,
        "index_dim": index_dim,
        "pool_size": 4,
        "cluster_block": 512,
        "buckets_per_block": 64,
        "max_bucket_keys": 16,
        "recommended_budget": int(args.recommended_budget),
        "head_slice_start": int(meta["lsa_slice_start"]),
        "head_slice_dim": int(meta["lsa_slice_dim"]),
        "full_attention_layers": full_layers,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    write_qlsa_sidecar(out_path, provenance, tensors)
    size_kb = out_path.stat().st_size / 1024.0
    print(
        f"[convert_lsa_sidecar] wrote {out_path} ({size_kb:.1f} KiB, "
        f"geometry_digest={geo_digest[:16]}...)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
