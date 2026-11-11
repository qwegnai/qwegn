#!/usr/bin/env python3
# Vectorized e-means clustering and codebook trainer (arXiv:2611.09501).
# Implements power-law (B, alpha0) scheduling, cosine EMA step annealing,
# casted-norm low-precision assignment, donor revival, and balanced Laguerre biases.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
import torch


class RegimeViolationError(ValueError):
    """Raised when e-means is invoked outside its valid n/k >= 8 regime."""


class EmeansRegime(Enum):
    IN_SAMPLE_STATIC = "in_sample_static"
    OUT_OF_SAMPLE_CALIB = "out_of_sample_calib"
    BALANCED_ROUTING = "balanced_routing"


@dataclass(frozen=True)
class EmeansCodebook:
    centroids: np.ndarray
    biases: np.ndarray | None
    train_mse: float
    dead_revivals: int


def _round_pow2(val: float) -> int:
    return 1 << int(math.floor(math.log2(max(val, 1.0)) + 0.5))


def emeans_schedule(
    n: int,
    k: int,
    epochs: float = 2.0,
    batch_mode: str | None = None,
) -> tuple[int, float]:
    """Power-law batch size B and coupled peak step alpha0 (arXiv:2611.09501 Table 2)."""
    if k <= 0 or n < 8 * k:
        raise RegimeViolationError(f"e-means requires n >= 8*k (got n={n}, k={k})")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    mode = batch_mode or ("valley" if n > 2_000_000 else "knee")
    if mode not in ("knee", "valley"):
        raise ValueError(f"batch_mode must be 'knee' or 'valley', got {mode!r}")
    coeff = 0.035 if mode == "knee" else 0.0044
    raw_b = coeff * (k ** 0.13) * (n ** 0.88) * (epochs ** 0.81)
    b = min(max(_round_pow2(raw_b), max(64, k)), _round_pow2(float(n)))
    b = min(b, n)
    alpha0 = min(0.824 * (k ** -0.46) * (b ** 0.90) * (n ** -0.45) * (epochs ** -0.11), 1.0)
    return int(b), float(alpha0)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "auto":
        return torch.float32
    raise ValueError(f"Unsupported compute_dtype {name!r}")


@torch.no_grad()
def _tiled_assign(
    xb: torch.Tensor,
    centroids: torch.Tensor,
    biases: torch.Tensor | None,
    compute_dtype: torch.dtype,
    tile_rows: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign rows to argmin_j (||c_hat_j||^2 - 2 <x_hat_i, c_hat_j> - b_j)."""
    b_rows = xb.shape[0]
    c_hat = centroids.to(dtype=compute_dtype)
    # Casted-centroid norm law (arXiv:2611.09501 §10): compute ||c_hat||^2 after cast.
    c_hat_f32 = c_hat.to(torch.float32)
    c_norm2 = (c_hat_f32 * c_hat_f32).sum(dim=1)
    if biases is not None:
        c_norm2 = c_norm2 - biases
    idx_out = torch.empty(b_rows, dtype=torch.int64, device=xb.device)
    min_term = torch.empty(b_rows, dtype=torch.float32, device=xb.device)
    for r0 in range(0, b_rows, tile_rows):
        r1 = min(r0 + tile_rows, b_rows)
        xt_hat = xb[r0:r1].to(dtype=compute_dtype)
        # Match Metal simdgroup_matrix<float,8,8> & TensorCore law: FP16/BF16 inputs, FP32 accumulate.
        dots = torch.matmul(xt_hat.to(torch.float32), c_hat_f32.T)
        scores = c_norm2.unsqueeze(0) - 2.0 * dots
        vals, idx = torch.min(scores, dim=1)
        idx_out[r0:r1] = idx
        min_term[r0:r1] = vals
    return idx_out, min_term


@torch.no_grad()
def _revive_dead_clusters(
    counts_ema: torch.Tensor,
    sums_ema: torch.Tensor,
    rng: np.random.Generator,
    tau: float = 0.1,
    delta: float = 1e-4,
) -> int:
    """Split high-mass donors into clusters with N_j < tau * mean(N)."""
    k, d = sums_ema.shape
    mean_n = float(counts_ema.mean().item())
    if mean_n <= 0.0:
        return 0
    dead_mask = counts_ema < (tau * mean_n)
    num_dead = int(dead_mask.sum().item())
    if num_dead == 0 or num_dead == k:
        return 0
    live_idx = torch.nonzero(~dead_mask, as_tuple=False).squeeze(1).cpu().numpy()
    dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(1).cpu().numpy()
    live_weights = np.maximum(counts_ema[live_idx].cpu().numpy().astype(np.float64), 0.0)
    w_sum = live_weights.sum()
    if w_sum <= 0.0:
        return 0
    probs = live_weights / w_sum
    take = min(num_dead, len(live_idx))
    donors = rng.choice(live_idx, size=take, replace=False, p=probs)
    targets = dead_idx[:take]
    signs = torch.ones(d, dtype=torch.float32, device=sums_ema.device)
    signs[1::2] = -1.0
    for j_dead, j_don in zip(targets, donors):
        n_don = counts_ema[j_don].clamp_min(1e-30)
        c_don = sums_ema[j_don] / n_don
        scale = float(torch.linalg.vector_norm(c_don).item()) / math.sqrt(max(d, 1))
        eps_vec = signs * delta * torch.maximum(c_don.abs(), torch.tensor( max(scale, 1e-6), device=sums_ema.device))
        c_hi = c_don + eps_vec
        c_lo = c_don - eps_vec
        half_n = 0.5 * n_don
        counts_ema[j_don] = half_n
        counts_ema[j_dead] = half_n
        sums_ema[j_don] = half_n * c_lo
        sums_ema[j_dead] = half_n * c_hi
    return int(take)


@torch.no_grad()
def fit_emeans(
    x: np.ndarray,
    k: int,
    regime: EmeansRegime = EmeansRegime.IN_SAMPLE_STATIC,
    epochs: float = 2.0,
    seed: int = 0,
    device: str = "cpu",
    compute_dtype: str = "auto",
    batch_mode: str | None = None,
    polish_steps: int = 20,
) -> EmeansCodebook:
    """Fit k centroids on x [n, d] using e-means (arXiv:2611.09501)."""
    x_np = np.asarray(x, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError(f"Expected 2D array [n, d], got shape {x_np.shape}")
    n, d = x_np.shape
    if d <= 0 or not np.isfinite(x_np).all():
        raise ValueError("Input array must have d > 0 and finite values")
    b, alpha0 = emeans_schedule(n=n, k=k, epochs=epochs, batch_mode=batch_mode)
    dev = torch.device(device)
    cdtype = _resolve_dtype(compute_dtype, dev)
    perm_rng = np.random.default_rng(seed)
    revival_rng = np.random.default_rng(seed + 1_000_003)

    init_ids = perm_rng.choice(n, size=k, replace=False)
    c0 = torch.from_numpy(x_np[init_ids].copy()).to(device=dev, dtype=torch.float32)
    counts_ema = torch.full((k,), 1e-10, device=dev, dtype=torch.float32)
    sums_ema = c0 * 1e-10
    biases = torch.zeros(k, device=dev, dtype=torch.float32) if regime == EmeansRegime.BALANCED_ROUTING else None

    steps_per_epoch = max(1, n // b)
    total_steps = max(1, int(round(epochs * n / b)))
    final_window_start = max(0, total_steps - steps_per_epoch)
    eps_floor = 0.0 if regime == EmeansRegime.BALANCED_ROUTING else 0.1

    last_counts = torch.zeros(k, device=dev, dtype=torch.float32)
    last_sums = torch.zeros((k, d), device=dev, dtype=torch.float32)
    ones_buf = torch.ones(b, device=dev, dtype=torch.float32)
    running_d: float | None = None
    revivals = 0
    perm = perm_rng.permutation(n)
    cursor = 0

    for step in range(total_steps):
        if cursor + b > n:
            perm = perm_rng.permutation(n)
            cursor = 0
        batch_ids = perm[cursor:cursor + b]
        cursor += b
        xb = torch.from_numpy(x_np[batch_ids]).to(device=dev, dtype=torch.float32)
        centroids = sums_ema / counts_ema.clamp_min(1e-30).unsqueeze(1)
        idx, min_term = _tiled_assign(xb, centroids, biases, cdtype)

        batch_counts = torch.zeros(k, device=dev, dtype=torch.float32)
        batch_sums = torch.zeros((k, d), device=dev, dtype=torch.float32)
        batch_counts.scatter_add_(0, idx, ones_buf[: xb.shape[0]])
        batch_sums.index_add_(0, idx, xb)

        cos_decay = 0.5 * (1.0 + math.cos(math.pi * step / max(total_steps, 1)))
        alpha_t = alpha0 * (eps_floor + (1.0 - eps_floor) * cos_decay)
        gamma_t = 1.0 - alpha_t
        counts_ema.mul_(gamma_t).add_(batch_counts, alpha=alpha_t)
        sums_ema.mul_(gamma_t).add_(batch_sums, alpha=alpha_t)

        if biases is not None:
            x_norm2 = (xb * xb).sum(dim=1)
            true_d = float((x_norm2 + min_term + biases[idx]).clamp_min(0.0).mean().item())
            running_d = true_d if running_d is None else (0.9 * running_d + 0.1 * true_d)
            target_c = xb.shape[0] / float(k)
            u_t = 0.02 * cos_decay
            biases.add_(((target_c - batch_counts) / target_c) * (u_t * running_d))
            biases.sub_(biases.mean())

        if step >= final_window_start:
            last_counts.add_(batch_counts)
            last_sums.add_(batch_sums)

        if (step + 1) % 5 == 0 and step + 1 < total_steps:
            revived = _revive_dead_clusters(counts_ema, sums_ema, revival_rng)
            revivals += revived
            if revived > 0 and step >= final_window_start:
                last_counts.zero_()
                last_sums.zero_()

    centroids_ema = sums_ema / counts_ema.clamp_min(1e-30).unsqueeze(1)
    if regime == EmeansRegime.IN_SAMPLE_STATIC:
        have_last = last_counts > 0
        final_c = torch.where(
            have_last.unsqueeze(1),
            last_sums / last_counts.clamp_min(1.0).unsqueeze(1),
            centroids_ema,
        )
    else:
        final_c = centroids_ema

    if biases is not None and polish_steps > 0:
        polish_b = min(n, max(b, 4_096))
        for p in range(polish_steps):
            pids = perm_rng.choice(n, size=polish_b, replace=False) if polish_b < n else np.arange(n)
            xb = torch.from_numpy(x_np[pids]).to(device=dev, dtype=torch.float32)
            idx, min_term = _tiled_assign(xb, final_c, biases, cdtype)
            p_counts = torch.bincount(idx, minlength=k).to(torch.float32)
            if running_d is None:
                running_d = float(((xb * xb).sum(dim=1) + min_term + biases[idx]).clamp_min(0.0).mean().item())
            target_c = polish_b / float(k)
            u_p = 0.04 * 0.5 * (1.0 + math.cos(math.pi * p / max(polish_steps, 1)))
            biases.add_(((target_c - p_counts) / target_c) * (u_p * running_d))
            biases.sub_(biases.mean())

    eval_n = min(n, 32_768)
    eval_x = torch.from_numpy(x_np[:eval_n]).to(device=dev, dtype=torch.float32)
    eval_idx, _ = _tiled_assign(eval_x, final_c, biases, cdtype)
    diff = eval_x - final_c[eval_idx]
    mse = float((diff * diff).sum(dim=1).mean().item())
    return EmeansCodebook(
        centroids=final_c.cpu().numpy().astype(np.float32),
        biases=biases.cpu().numpy().astype(np.float32) if biases is not None else None,
        train_mse=mse,
        dead_revivals=revivals,
    )


@torch.no_grad()
def assign_emeans(
    x: np.ndarray,
    codebook: EmeansCodebook,
    device: str = "cpu",
    compute_dtype: str = "float32",
    tile_rows: int = 32_768,
) -> np.ndarray:
    """Assign rows [n, d] to codebook centroids (applying Laguerre biases when present)."""
    x_np = np.asarray(x, dtype=np.float32)
    dev = torch.device(device)
    cdtype = _resolve_dtype(compute_dtype, dev)
    c_t = torch.from_numpy(codebook.centroids).to(device=dev, dtype=torch.float32)
    b_t = torch.from_numpy(codebook.biases).to(device=dev, dtype=torch.float32) if codebook.biases is not None else None
    out = np.empty(x_np.shape[0], dtype=np.int64)
    for r0 in range(0, x_np.shape[0], tile_rows):
        r1 = min(r0 + tile_rows, x_np.shape[0])
        xb = torch.from_numpy(x_np[r0:r1]).to(device=dev, dtype=torch.float32)
        idx, _ = _tiled_assign(xb, c_t, b_t, cdtype, tile_rows=tile_rows)
        out[r0:r1] = idx.cpu().numpy()
    return out
