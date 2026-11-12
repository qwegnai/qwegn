# QNGRAM v3 VQ4_G32 (4D 256-centroid e-means, 50 B/row = 2.50 bpw) trainer,
# MPS/CPU 2-step ALS encoder, validator, and exact IEEE-754 float32 decoder.
from __future__ import annotations

import numpy as np

NG_GROUP = 32
NG_VQ4_CB_OFF = 1024
NG_VQ4_CB_BYTES = 2048
NG_VQ4_CB_SCALARS = 1024
_F16_MAX = 65504.0


def canonical_vq4_codebook() -> np.ndarray:
    rng = np.random.default_rng(20261104)
    cb = np.zeros((256, 4), dtype=np.float32)
    pts = rng.standard_normal((255, 4)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=-1, keepdims=True).clip(1e-8)
    radii = np.linspace(0.20, 0.99, 255, dtype=np.float32)[:, None]
    cb[1:] = (pts / norms) * radii
    cb16 = cb.astype(np.float16)
    cb16[0] = np.float16(0.0)
    return cb16


def train_vq4_codebook(rows_f32: np.ndarray, seed: int = 0, device: str = "cpu") -> np.ndarray:
    from emeans import EmeansRegime, fit_emeans

    groups = rows_f32.shape[1] // NG_GROUP
    quads = NG_GROUP // 4
    xg = rows_f32.reshape(-1, groups, quads, 4)
    quad_norms = np.linalg.norm(xg, axis=-1, keepdims=True)
    gmax = quad_norms.max(axis=2, keepdims=True)
    live_mask = gmax.squeeze(-1).squeeze(-1) > 0
    if int(live_mask.sum()) * quads < 8 * 256:
        return canonical_vq4_codebook()
    unit_quads = (xg[live_mask] / gmax[live_mask]).reshape(-1, 4)
    cb = fit_emeans(
        unit_quads,
        k=256,
        regime=EmeansRegime.IN_SAMPLE_STATIC,
        epochs=2.5,
        seed=seed,
        device=device,
        batch_mode="valley",
    )
    lut = cb.centroids.copy()
    zero_slot = int(np.argmin((lut * lut).sum(axis=1)))
    lut[zero_slot] = lut[0]
    lut[0] = 0.0
    lut16 = lut.astype(np.float16)
    lut16[0] = np.float16(0.0)
    return lut16


def validate_vq4_codebook_bytes(cb_bytes: bytes) -> None:
    if len(cb_bytes) != NG_VQ4_CB_BYTES:
        raise ValueError("ngram.bin: v3 vq4 codebook slice short")
    cb = np.frombuffer(cb_bytes, dtype="<f2").reshape(256, 4)
    if not np.isfinite(cb).all():
        raise ValueError("ngram.bin: non-finite value in v3 vq4 codebook")
    raw_u16 = np.frombuffer(cb_bytes[:8], dtype="<u2")
    if any(int(u) != 0 for u in raw_u16):
        raise ValueError("ngram.bin: v3 vq4 codebook[0] must be canonical +0.0")
    if not (np.abs(cb[1:]) > 0).any():
        raise ValueError("ngram.bin: v3 vq4 codebook is all zero")


def encode_vq4(rows_f32: np.ndarray, codebook_f16: np.ndarray, device: str = "cpu") -> bytes:
    import torch

    n, dim = rows_f32.shape
    groups = dim // NG_GROUP
    quads = NG_GROUP // 4
    dev = (
        torch.device("mps")
        if (device == "mps" and torch.backends.mps.is_available())
        else torch.device("cpu")
    )
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(rows_f32, dtype=np.float32)).to(device=dev)
        xg = xt.reshape(n, groups, quads, 4)
        s0 = torch.linalg.vector_norm(xg, dim=-1).amax(dim=-1)
        if bool((s0 > _F16_MAX).any().item()):
            raise ValueError("ngram.bin: a row's scale does not fit f16")
        cb = torch.from_numpy(
            np.ascontiguousarray(codebook_f16, dtype="<f2").astype(np.float32)
        ).to(device=dev)
        cb_n2 = (cb * cb).sum(dim=-1)
        scale16 = s0.to(torch.float16)
        xg_flat = xt.reshape(n, groups, NG_GROUP)
        total_quads = n * groups * quads
        bs = 524_288
        idx_buf = torch.empty(total_quads, device=dev, dtype=torch.int64)
        for _ in range(2):
            s_f32 = scale16.to(torch.float32)
            live = s_f32 > 0
            s_safe = torch.where(live, s_f32, torch.ones_like(s_f32))
            normed = (xg / s_safe[:, :, None, None]).reshape(total_quads, 4)
            for i in range(0, total_quads, bs):
                ub = normed[i : i + bs]
                idx_buf[i : i + bs] = (cb_n2[None, :] - 2.0 * (ub @ cb.t())).argmin(dim=-1)
            idx = torch.where(live[:, :, None], idx_buf.reshape(n, groups, quads), 0)
            u = cb[idx].reshape(n, groups, NG_GROUP)
            denom = (u * u).sum(dim=-1)
            s_ls = torch.where(
                denom > 0,
                (xg_flat * u).sum(dim=-1) / denom.clamp(min=1e-30),
                torch.zeros_like(denom),
            ).clamp(min=0.0, max=_F16_MAX)
            scale16 = torch.where(s0 > 0, s_ls.to(torch.float16), torch.zeros_like(scale16))
        s_final_f32 = scale16.to(torch.float32)
        live_final = s_final_f32 > 0
        s_final = torch.where(live_final, s_final_f32, torch.ones_like(s_final_f32))
        normed = (xg / s_final[:, :, None, None]).reshape(total_quads, 4)
        for i in range(0, total_quads, bs):
            ub = normed[i : i + bs]
            idx_buf[i : i + bs] = (cb_n2[None, :] - 2.0 * (ub @ cb.t())).argmin(dim=-1)
        codes_t = (
            torch.where(live_final[:, :, None], idx_buf.reshape(n, groups, quads), 0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        sc_u8 = scale16.cpu().numpy().view(np.uint8).reshape(n, groups, 2)
        out = np.empty((n, groups, quads + 2), dtype=np.uint8)
        out[..., :quads] = codes_t
        out[..., quads:] = sc_u8
        return out.tobytes()


def decode_vq4_rows(raw_bytes: bytes, codebook_f16, rows: int, head_dim: int = 160) -> np.ndarray:
    groups = head_dim // NG_GROUP
    quads = NG_GROUP // 4
    lut32 = np.asarray(codebook_f16, dtype="<f2").astype(np.float32).reshape(256, 4)
    enc = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rows, groups, quads + 2)
    codes = enc[:, :, :quads]
    scales = np.ascontiguousarray(enc[:, :, quads:]).view("<f2").reshape(rows, groups).astype(np.float32)
    vals = lut32[codes] * scales[:, :, None, None]
    return vals.reshape(rows, head_dim)
