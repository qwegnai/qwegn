"""Quantize-then-dequantize HQ weights in torch, without packing bytes.

These functions reproduce the reconstruction that `hq_blocks.quant_hq_blocks`
followed by `hq_blocks.dequant_hq_blocks` yields, so quantization error can be
measured on many candidate formats without materializing whole packs. The
equivalence is asserted against the byte-level writers in test_hq_fakequant.py.
"""
from __future__ import annotations

import math

import torch

from hq4 import BIAS_LAW, MAX_SCALE, SCALE_DIVISOR
from hq_blocks import GROUP, VALID_BITS, parse_quant_spec, resolve_group_size

_INV_SQRT32 = 1.0 / math.sqrt(32.0)
_SHRINKS = (1.00, 0.96, 0.92, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70)


def fwht32(t: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over consecutive groups of 32 (self-inverse)."""
    shape = t.shape
    h = 1
    while h < 32:
        t = t.reshape(-1, 32 // (2 * h), 2, h)
        t = torch.cat((t[:, :, 0, :] + t[:, :, 1, :], t[:, :, 0, :] - t[:, :, 1, :]), dim=-1)
        h *= 2
    return t.reshape(shape) * _INV_SQRT32


def _m5_rot(w: torch.Tensor) -> torch.Tensor:
    g = w.reshape(*w.shape[:-1], -1, GROUP)
    amax = g.abs().amax(dim=-1)
    scale = torch.minimum(amax / SCALE_DIVISOR, torch.tensor(MAX_SCALE, device=w.device)).half().float()
    bias = (BIAS_LAW * scale).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale)).unsqueeze(-1)
    q = torch.floor((g - bias.unsqueeze(-1)) / safe - 1023.5).clamp(0, 15)
    q = torch.where(scale.unsqueeze(-1) > 0, q, torch.full_like(q, 8.0))
    return ((q + 1024.0) * scale.unsqueeze(-1) + bias.unsqueeze(-1)).reshape(w.shape)


def _unified_rot(w: torch.Tensor, bits: int, group: int, mse_clip: bool) -> torch.Tensor:
    qmax = float((1 << bits) - 1)
    g = w.reshape(*w.shape[:-1], -1, group)
    w_min = g.amin(dim=-1)
    w_max = g.amax(dim=-1)
    scale = ((w_max - w_min) / qmax).half()
    bias = w_min.half()
    if mse_clip:
        mid = 0.5 * (w_max + w_min)
        half_span = 0.5 * (w_max - w_min)
        best = torch.full_like(w_min, float("inf"))
        for shrink in _SHRINKS:
            c_min = mid - shrink * half_span
            sc16 = ((mid + shrink * half_span - c_min) / qmax).half()
            bi16 = c_min.half()
            sc = sc16.float().unsqueeze(-1)
            bi = bi16.float().unsqueeze(-1)
            safe = torch.where(sc > 0, sc, torch.ones_like(sc))
            q = torch.where(sc > 0, torch.round((g - bi) / safe).clamp(0, qmax), torch.zeros_like(g))
            err = ((g - (q * sc + bi)) ** 2).mean(dim=-1)
            better = err < best
            best = torch.where(better, err, best)
            scale = torch.where(better, sc16, scale)
            bias = torch.where(better, bi16, bias)
    sc = scale.float().unsqueeze(-1)
    bi = bias.float().unsqueeze(-1)
    safe = torch.where(sc > 0, sc, torch.ones_like(sc))
    q = torch.where(sc > 0, torch.round((g - bi) / safe).clamp(0, qmax), torch.zeros_like(g))
    return (q * sc + bi).reshape(w.shape)


def fake_quant(w: torch.Tensor, spec: str, chunk_rows: int = 1 << 15) -> torch.Tensor:
    """Return the float32 weight that a pack written with `spec` would reconstruct.

    `w` is [..., K] with the HQ input dimension last; `spec` is `bf16` or an
    `hq_blocks` quantization spec such as `hq4_g32_m5` or `hq3_g64`.
    """
    if spec == "bf16":
        return w.to(torch.bfloat16).float()
    bits, requested, is_m5 = parse_quant_spec(spec)
    if bits not in VALID_BITS:
        raise ValueError(f"unsupported bits in {spec}")
    k = w.shape[-1]
    group = resolve_group_size(k, requested)
    if is_m5 and (bits != 4 or group != 32):
        raise ValueError(f"{spec}: M5 bias law exists only for HQ4_G32")
    if not bool(torch.isfinite(w).all()):
        raise ValueError("fake_quant rejects non-finite weights")
    flat = w.reshape(-1, k)
    out = torch.empty(flat.shape, dtype=torch.float32, device=w.device)
    mse_clip = bits in (2, 3)
    for r0 in range(0, flat.shape[0], chunk_rows):
        rot = fwht32(flat[r0:r0 + chunk_rows].float())
        rec = _m5_rot(rot) if is_m5 else _unified_rot(rot, bits, group, mse_clip)
        out[r0:r0 + chunk_rows] = fwht32(rec)
    return out.reshape(w.shape)
