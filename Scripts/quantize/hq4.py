"""The HQ4_G32_M5 block layout — one source of truth for every writer.

A group of 32 weights becomes exactly 20 bytes:
    bytes  0..2   f16 scale
    bytes  2..4   f16 bias  (= -1032 * scale)
    bytes  4..20  uint32 qs[4], 32 nibbles

Every writer that emits Block_HQ4_G32 — the model packer and the QNGRAM
table builder — must use THIS module, or the two layouts drift apart and the
runtime reads the wrong weights.
"""
import math

import numpy as np

GROUP = 32
BLOCK_BYTES = 20
INV_SQRT32 = np.float32(1.0 / math.sqrt(32.0))

# The bias law and the scale clamp of the format.
BIAS_LAW = -1032.0
SCALE_DIVISOR = 7.5
MAX_SCALE = 63.46875


def fwht32_ortho(x: np.ndarray) -> np.ndarray:
    orig_shape = x.shape
    assert orig_shape[-1] % 32 == 0, f"last dim {orig_shape[-1]} must be divisible by 32"
    try:
        import torch
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        h = 1
        while h < 32:
            t = t.reshape(-1, 32 // (2 * h), 2, h)
            u = t[:, :, 0, :]
            v = t[:, :, 1, :]
            t = torch.cat((u + v, u - v), dim=-1)
            h *= 2
        t = t.reshape(orig_shape) * float(INV_SQRT32)
        return t.numpy()
    except ImportError:
        y = np.array(x, dtype=np.float32, copy=True)
        h = 1
        while h < 32:
            y = y.reshape(-1, 32 // (2 * h), 2, h)
            u = y[:, :, 0, :].copy()
            v = y[:, :, 1, :].copy()
            y[:, :, 0, :] = u + v
            y[:, :, 1, :] = u - v
            h *= 2
        return (y.reshape(orig_shape) * INV_SQRT32).astype(np.float32)


def quant_hq4_blocks(w, rotate: bool = True) -> np.ndarray:
    """Write Block_HQ4_G32 blocks for a 2-D tensor.

    `rotate=True` applies the orthonormal Walsh-Hadamard transform within
    each group of 32 before quantising. That is the form the **weight pack**
    uses: a weight is multiplied by a matrix, so rotating both sides leaves
    the product alone while flattening the outliers.

    `rotate=False` quantises the values as they are. That is the form the
    **n-gram table** uses: a looked-up row is *added* to the residual
    stream, never multiplied by a weight matrix, so there is nothing to
    rotate it against and the rotation would have to be undone on every
    read. The block layout is identical either way.
    """
    import torch
    if isinstance(w, torch.Tensor):
        if w.ndim != 2 or w.shape[0] <= 0 or w.shape[1] <= 0 or w.shape[1] % GROUP:
            raise ValueError(f"HQ4_G32_M5 requires a non-empty 2-D tensor with K divisible by {GROUP}")
        if not torch.isfinite(w).all():
            raise ValueError("HQ4_G32_M5 source weights must be finite")
        w_t = w
    else:
        w_arr = np.asarray(w, dtype=np.float32)
        if w_arr.ndim != 2 or w_arr.shape[0] <= 0 or w_arr.shape[1] <= 0 or w_arr.shape[1] % GROUP:
            raise ValueError(f"HQ4_G32_M5 requires a non-empty 2-D tensor with K divisible by {GROUP}")
        if not np.isfinite(w_arr).all():
            raise ValueError("HQ4_G32_M5 source weights must be finite")
        w_t = torch.from_numpy(np.ascontiguousarray(w_arr))
    N, K = w_t.shape
    G = K // GROUP
    inv_sqrt32 = float(INV_SQRT32)
    max_s = torch.tensor(MAX_SCALE, dtype=torch.float32)
    blocks = np.empty((N, G, BLOCK_BYTES), dtype=np.uint8)
    step = 8192
    for r0 in range(0, N, step):
        r1 = min(r0 + step, N)
        rn = r1 - r0
        t = w_t[r0:r1].float() if w_t.dtype != torch.float32 else w_t[r0:r1]
        if rotate:
            h = 1
            while h < 32:
                t = t.reshape(-1, 32 // (2 * h), 2, h)
                u = t[:, :, 0, :]
                v = t[:, :, 1, :]
                t = torch.cat((u + v, u - v), dim=-1)
                h *= 2
            w_rot = t.reshape(rn, G, GROUP) * inv_sqrt32
        else:
            w_rot = t.reshape(rn, G, GROUP)
        amax = w_rot.abs().amax(dim=-1)
        scale_f16 = torch.minimum(amax / SCALE_DIVISOR, max_s).to(torch.float16)
        exact_scale = scale_f16.float()
        bias_f16 = (BIAS_LAW * exact_scale).to(torch.float16)
        exact_bias = bias_f16.float()
        safe_scale = torch.where(exact_scale > 0, exact_scale, torch.ones_like(exact_scale)).unsqueeze(-1)
        q_float = torch.floor((w_rot - exact_bias.unsqueeze(-1)) / safe_scale - 1023.5)
        q = torch.where(exact_scale.unsqueeze(-1) > 0, torch.clamp(q_float, 0, 15),
                        torch.full_like(q_float, 8.0)).to(torch.int32)
        q8 = q.reshape(rn, G, 4, 8)
        qs = (
            (q8[..., 0] << 0)  | (q8[..., 1] << 16) |
            (q8[..., 2] << 4)  | (q8[..., 3] << 20) |
            (q8[..., 4] << 8)  | (q8[..., 5] << 24) |
            (q8[..., 6] << 12) | (q8[..., 7] << 28)
        )
        sub = blocks[r0:r1]
        sub[..., 0:2] = scale_f16.numpy().view(np.uint8).reshape(rn, G, 2)
        sub[..., 2:4] = bias_f16.numpy().view(np.uint8).reshape(rn, G, 2)
        sub[..., 4:20] = qs.numpy().view(np.uint8).reshape(rn, G, 16)
    return blocks


def quant_hq4_g32_m5(w) -> np.ndarray:
    """The weight-pack form: rotate, then write blocks."""
    return quant_hq4_blocks(w, rotate=True)


def dequant_hq4_g32_m5_rot(blocks: np.ndarray, f16_round: bool = False) -> np.ndarray:
    N, G, B = blocks.shape
    assert B == BLOCK_BYTES, f"expected {BLOCK_BYTES} bytes per block, got {B}"
    scale = np.ascontiguousarray(blocks[..., 0:2]).view(np.float16).astype(np.float32)
    bias = np.ascontiguousarray(blocks[..., 2:4]).view(np.float16).astype(np.float32)
    qs = np.ascontiguousarray(blocks[..., 4:20]).view(np.uint32)

    q = np.empty((N, G, GROUP), dtype=np.float32)
    for j in range(4):
        b = 8 * j
        raw = qs[..., j]
        q[..., b + 0] = (raw >> 0) & 0xF
        q[..., b + 1] = (raw >> 16) & 0xF
        q[..., b + 2] = (raw >> 4) & 0xF
        q[..., b + 3] = (raw >> 20) & 0xF
        q[..., b + 4] = (raw >> 8) & 0xF
        q[..., b + 5] = (raw >> 24) & 0xF
        q[..., b + 6] = (raw >> 12) & 0xF
        q[..., b + 7] = (raw >> 28) & 0xF

    w_rot = (q + np.float32(1024.0)) * scale + bias
    if f16_round:
        w_rot = w_rot.astype(np.float16).astype(np.float32)
    return w_rot.reshape(N, G * GROUP)


def dequant_hq4_g32_m5(blocks: np.ndarray) -> np.ndarray:
    return fwht32_ortho(dequant_hq4_g32_m5_rot(blocks, f16_round=False))
