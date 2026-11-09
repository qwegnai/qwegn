# Unified HQ{2,3,4,6,8} block quantizer and dequantizer for G{32,64,128,256}.
# Preserves HQ4_G32_M5 byte-for-byte and implements asymmetric rotated
# [min, max] continuous-bitstream blocks for HQ{2,3,4,6,8}_G{32,64,128,256}.
from __future__ import annotations

import math
import numpy as np

from hq4 import (
    BLOCK_BYTES,
    GROUP,
    INV_SQRT32,
    dequant_hq4_g32_m5,
    dequant_hq4_g32_m5_rot,
    quant_hq4_blocks,
)

VALID_BITS = (2, 3, 4, 6, 8)
VALID_GROUPS = (32, 64, 128, 256)
_HQ_PREFIXES = ("hq2", "hq3", "hq4", "hq6", "hq8")


# Dequantize block-scaled FP8 (e.g. F8_E4M3 + [ceil(N/128), ceil(K/128)] weight_scale_inv) to float32 tensor.
def dequant_fp8_blockwise(w_fp8, scale_inv, block_size: int = 128, device=None):
    import torch
    if not isinstance(w_fp8, torch.Tensor):
        w_fp8 = torch.as_tensor(w_fp8)
    if not isinstance(scale_inv, torch.Tensor):
        scale_inv = torch.as_tensor(scale_inv)
    target_dev = device or (scale_inv.device if scale_inv.device.type != "mps" else torch.device("cpu"))
    # PyTorch MPS backend does not support float8_e4m3fn cast directly; cast to float32 on CPU first.
    wf = w_fp8.cpu().to(torch.float32)
    if device is not None:
        wf = wf.to(device=device)
    sf = scale_inv.to(device=wf.device, dtype=torch.float32)
    if wf.ndim == 2:
        n, k = wf.shape
        sn, sk = sf.shape
        if sn * block_size == n and sk * block_size == k:
            return (wf.reshape(sn, block_size, sk, block_size) * sf[:, None, :, None]).reshape(n, k)
        s_exp = sf[:, None, :, None].expand(sn, block_size, sk, block_size).reshape(sn * block_size, sk * block_size)
        return wf * s_exp[:n, :k]
    if wf.ndim == 3:
        e, n, k = wf.shape
        _, sn, sk = sf.shape
        if sn * block_size == n and sk * block_size == k:
            return (wf.reshape(e, sn, block_size, sk, block_size) * sf[:, :, None, :, None]).reshape(e, n, k)
        s_exp = sf[:, :, None, :, None].expand(e, sn, block_size, sk, block_size).reshape(e, sn * block_size, sk * block_size)
        return wf * s_exp[:, :n, :k]
    raise ValueError(f"Unsupported FP8 tensor rank {wf.ndim} with shape {tuple(wf.shape)}")


# Apply 32-element orthonormal Fast Walsh-Hadamard Transform on a torch Tensor.
def _fwht32_torch(t):
    import torch
    orig_shape = t.shape
    h = 1
    while h < 32:
        t = t.reshape(-1, 32 // (2 * h), 2, h)
        u = t[:, :, 0, :]
        v = t[:, :, 1, :]
        t = torch.cat((u + v, u - v), dim=-1)
        h *= 2
    return t.reshape(orig_shape) * float(INV_SQRT32)


# Parse quantization spec string into (bits, group, is_m5).
def parse_quant_spec(spec: str, default_group: int | None = None) -> tuple[int, int, bool]:
    s = spec.strip().lower()
    if default_group is not None and default_group not in VALID_GROUPS:
        raise ValueError(f"Unsupported group size {default_group}; expected {VALID_GROUPS}")
    if s in ("hq4_g32_m5", "q4g"):
        g = default_group if default_group is not None else 32
        return (4, g, g == 32)
    if s in _HQ_PREFIXES:
        b = int(s[2:])
        g = default_group if default_group is not None else 32
        return (b, g, b == 4 and g == 32)
    is_m5 = False
    if s.endswith("_m5"):
        is_m5 = True
        s = s[:-3]
    parts = s.split("_")
    if len(parts) == 2 and parts[0] in _HQ_PREFIXES and parts[1].startswith("g"):
        b = int(parts[0][2:])
        try:
            g = int(parts[1][1:])
        except ValueError as exc:
            raise ValueError(f"Invalid group in quant spec: {spec!r}") from exc
        if g not in VALID_GROUPS:
            raise ValueError(f"Unsupported group size {g} in {spec!r}; expected {VALID_GROUPS}")
        if is_m5 and (b != 4 or g != 32):
            raise ValueError(f"_m5 magic format is only supported for hq4_g32, got {spec!r}")
        return (b, g, is_m5)
    raise ValueError(f"Invalid quant spec {spec!r}; expected e.g. hq2_g128, hq3_g128, hq4_g32_m5, hq6_g32, hq8_g128")


# Format (bits, group, is_m5) back into canonical spec name.
def format_quant_spec(bits: int, group: int, is_m5: bool = False) -> str:
    if bits == 4 and group == 32 and is_m5:
        return "hq4_g32_m5"
    return f"hq{bits}_g{group}"


# Clamp requested group size to largest divisor of k in VALID_GROUPS.
def resolve_group_size(k: int, requested_group: int) -> int:
    if k <= 0 or (k % 32) != 0:
        raise ValueError(f"Inner dimension K={k} must be a positive multiple of 32")
    if requested_group not in VALID_GROUPS:
        raise ValueError(f"Unsupported group size {requested_group}; expected {VALID_GROUPS}")
    eff = math.gcd(k, requested_group)
    if eff not in VALID_GROUPS:
        raise ValueError(f"Resolved group size {eff} for K={k}, group={requested_group} not in {VALID_GROUPS}")
    return eff


# Byte size of a single quantized group block (4B header + payload).
def hq_block_bytes(bits: int, group: int) -> int:
    if bits not in VALID_BITS or group not in VALID_GROUPS:
        raise ValueError(f"Unsupported (bits={bits}, group={group})")
    return (group * bits) // 8 + 4


# Pack unsigned integers q [..., G] into continuous little-endian bit stream.
def _pack_bitstream(q, bits: int):
    import torch
    if bits == 2:
        v0 = q[..., 0::4] & 0x03
        v1 = q[..., 1::4] & 0x03
        v2 = q[..., 2::4] & 0x03
        v3 = q[..., 3::4] & 0x03
        return (v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)).to(torch.uint8)
    if bits == 3:
        qi = q.to(torch.int32) & 0x07
        w24 = (
            qi[..., 0::8]
            | (qi[..., 1::8] << 3)
            | (qi[..., 2::8] << 6)
            | (qi[..., 3::8] << 9)
            | (qi[..., 4::8] << 12)
            | (qi[..., 5::8] << 15)
            | (qi[..., 6::8] << 18)
            | (qi[..., 7::8] << 21)
        )
        b0 = (w24 & 0xFF).to(torch.uint8)
        b1 = ((w24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((w24 >> 16) & 0xFF).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(*q.shape[:-1], (q.shape[-1] * 3) // 8)
    if bits == 4:
        return (q[..., 0::2] & 0x0F) | ((q[..., 1::2] & 0x0F) << 4)
    if bits == 6:
        v0 = q[..., 0::4] & 0x3F
        v1 = q[..., 1::4] & 0x3F
        v2 = q[..., 2::4] & 0x3F
        v3 = q[..., 3::4] & 0x3F
        b0 = (v0 | ((v1 & 0x03) << 6)).to(torch.uint8)
        b1 = ((v1 >> 2) | ((v2 & 0x0F) << 4)).to(torch.uint8)
        b2 = ((v2 >> 4) | (v3 << 2)).to(torch.uint8)
        return torch.stack([b0, b1, b2], dim=-1).reshape(*q.shape[:-1], (q.shape[-1] * 3) // 4)
    if bits == 8:
        return q.to(torch.uint8)
    raise ValueError(f"Unsupported bits={bits}")


# Unpack continuous little-endian bit stream payload [..., G*bits/8] to float32 [..., G].
def _unpack_bitstream(payload: np.ndarray, bits: int, group: int) -> np.ndarray:
    if bits == 2:
        v0 = payload & 0x03
        v1 = (payload >> 2) & 0x03
        v2 = (payload >> 4) & 0x03
        v3 = (payload >> 6) & 0x03
        return np.stack([v0, v1, v2, v3], axis=-1).reshape(*payload.shape[:-1], group).astype(np.float32)
    if bits == 3:
        b0 = payload[..., 0::3].astype(np.uint32)
        b1 = payload[..., 1::3].astype(np.uint32)
        b2 = payload[..., 2::3].astype(np.uint32)
        w24 = b0 | (b1 << 8) | (b2 << 16)
        vals = [((w24 >> (3 * j)) & 0x07) for j in range(8)]
        return np.stack(vals, axis=-1).reshape(*payload.shape[:-1], group).astype(np.float32)
    if bits == 4:
        lo = payload & 0x0F
        hi = (payload >> 4) & 0x0F
        return np.stack([lo, hi], axis=-1).reshape(*payload.shape[:-1], group).astype(np.float32)
    if bits == 6:
        b0 = payload[..., 0::3]
        b1 = payload[..., 1::3]
        b2 = payload[..., 2::3]
        v0 = b0 & 0x3F
        v1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
        v2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
        v3 = (b2 >> 2) & 0x3F
        return np.stack([v0, v1, v2, v3], axis=-1).reshape(*payload.shape[:-1], group).astype(np.float32)
    if bits == 8:
        return payload.astype(np.float32)
    raise ValueError(f"Unsupported bits={bits}")


_MPS_QUANT_LIB = None
_MPS_QUANT_FAILED = False
_MPS_QUANT_SRC = """
#include <metal_stdlib>
using namespace metal;

constant float SHRINKS[10] = {1.00f, 0.96f, 0.92f, 0.88f, 0.85f, 0.82f, 0.79f, 0.76f, 0.73f, 0.70f};
constant float INV_SQRT32 = 0.17677669529663689f;
constant float BIAS_LAW = -1032.0f;
constant float SCALE_DIVISOR = 7.5f;
constant float MAX_SCALE = 63.46875f;

kernel void quant_hq4_m5_simd(
    device const float* w [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant int& total_groups [[buffer(2)]],
    constant int& do_rotate [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint gid = tid >> 5;
    if ((int)gid >= total_groups) return;
    float v = w[(size_t)gid * 32 + (size_t)lane];
    if (do_rotate != 0) {
        for (uint h = 1; h < 32; h <<= 1) {
            float partner = simd_shuffle_xor(v, ushort(h));
            v = ((lane & h) == 0) ? (v + partner) : (partner - v);
        }
        v *= INV_SQRT32;
    }
    float amax = simd_max(fabs(v));
    half scale_f16 = half(fmin(amax / SCALE_DIVISOR, MAX_SCALE));
    float exact_scale = float(scale_f16);
    half bias_f16 = half(BIAS_LAW * exact_scale);
    float exact_bias = float(bias_f16);
    uint q = 8u;
    if (exact_scale > 0.0f) {
        float qf = floor((v - exact_bias) / exact_scale - 1023.5f);
        q = uint(clamp(qf, 0.0f, 15.0f));
    }
    device uchar* gout = out + (size_t)gid * 20;
    if (lane == 0) {
        ushort s_bits = as_type<ushort>(scale_f16);
        ushort b_bits = as_type<ushort>(bias_f16);
        gout[0] = uchar(s_bits & 0xFF);
        gout[1] = uchar(s_bits >> 8);
        gout[2] = uchar(b_bits & 0xFF);
        gout[3] = uchar(b_bits >> 8);
    }
    uint q0 = q & 0xFu;
    uint w32 = 0u;
    for (uint j = 0; j < 8; ++j) {
        uint qj = simd_shuffle(q0, ushort((lane & ~7u) | j));
        uint shift = ((j >> 1) * 4u) + ((j & 1u) * 16u);
        w32 |= (qj << shift);
    }
    if ((lane & 7u) == 0u) {
        device uint* qs = reinterpret_cast<device uint*>(gout + 4);
        qs[lane >> 3] = w32;
    }
}

kernel void quant_hq_simd(
    device const float* w [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant int& total_groups [[buffer(2)]],
    constant int& bits [[buffer(3)]],
    constant int& group_size [[buffer(4)]],
    constant int& do_rotate [[buffer(5)]],
    constant int& do_mse [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint gid = tid >> 5;
    if ((int)gid >= total_groups) return;

    int chunks = group_size >> 5;
    int bbytes = ((group_size * bits) >> 3) + 4;
    device const float* gw = w + (size_t)gid * (size_t)group_size;
    device uchar* gout = out + (size_t)gid * (size_t)bbytes;

    float vals[8];
    float lmin = INFINITY;
    float lmax = -INFINITY;
    for (int c = 0; c < chunks; ++c) {
        float v = gw[(c << 5) + (int)lane];
        if (do_rotate != 0) {
            for (uint h = 1; h < 32; h <<= 1) {
                float partner = simd_shuffle_xor(v, ushort(h));
                v = ((lane & h) == 0) ? (v + partner) : (partner - v);
            }
            v *= INV_SQRT32;
        }
        vals[c] = v;
        lmin = fmin(lmin, v);
        lmax = fmax(lmax, v);
    }
    float w_min = simd_min(lmin);
    float w_max = simd_max(lmax);
    float qmax = float((1 << bits) - 1);

    half best_scale = half((w_max - w_min) / qmax);
    half best_bias = half(w_min);
    if (do_mse != 0) {
        float mid = 0.5f * (w_max + w_min);
        float half_span = 0.5f * (w_max - w_min);
        float best_err = INFINITY;
        for (int s = 0; s < 10; ++s) {
            float sh = SHRINKS[s];
            float c_min = mid - sh * half_span;
            float c_max = mid + sh * half_span;
            half sc16 = half((c_max - c_min) / qmax);
            half bi16 = half(c_min);
            float sc32 = float(sc16);
            float bi32 = float(bi16);
            float safe_sc = (sc32 > 0.0f) ? sc32 : 1.0f;
            float lerr = 0.0f;
            for (int c = 0; c < chunks; ++c) {
                float q = (sc32 > 0.0f) ? clamp(rint((vals[c] - bi32) / safe_sc), 0.0f, qmax) : 0.0f;
                float recon = q * sc32 + bi32;
                float d = vals[c] - recon;
                lerr += d * d;
            }
            float err = simd_sum(lerr);
            if (err < best_err) {
                best_err = err;
                best_scale = sc16;
                best_bias = bi16;
            }
        }
    }

    if (lane == 0) {
        ushort s_bits = as_type<ushort>(best_scale);
        ushort b_bits = as_type<ushort>(best_bias);
        gout[0] = uchar(s_bits & 0xFF);
        gout[1] = uchar(s_bits >> 8);
        gout[2] = uchar(b_bits & 0xFF);
        gout[3] = uchar(b_bits >> 8);
    }

    float sc32 = float(best_scale);
    float bi32 = float(best_bias);
    float safe_sc = (sc32 > 0.0f) ? sc32 : 1.0f;
    device uchar* payload = gout + 4;

    for (int c = 0; c < chunks; ++c) {
        uint q = (sc32 > 0.0f) ? uint(clamp(rint((vals[c] - bi32) / safe_sc), 0.0f, qmax)) : 0u;
        if (bits == 2) {
            uint q0 = q & 0x3u;
            uint q1 = simd_shuffle_xor(q0, 1);
            uint q2 = simd_shuffle_xor(q0, 2);
            uint q3 = simd_shuffle_xor(q0, 3);
            if ((lane & 3u) == 0u) {
                payload[c * 8 + (lane >> 2)] = uchar(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
            }
        } else if (bits == 3) {
            uint q0 = q & 0x7u;
            uint w24 = 0u;
            for (uint j = 0; j < 8; ++j) {
                uint qj = simd_shuffle(q0, ushort((lane & ~7u) | j));
                w24 |= (qj << (3u * j));
            }
            if ((lane & 7u) == 0u) {
                int off = c * 12 + int(lane >> 3) * 3;
                payload[off + 0] = uchar(w24 & 0xFFu);
                payload[off + 1] = uchar((w24 >> 8) & 0xFFu);
                payload[off + 2] = uchar((w24 >> 16) & 0xFFu);
            }
        } else if (bits == 4) {
            uint q0 = q & 0xFu;
            uint q1 = simd_shuffle_xor(q0, 1);
            if ((lane & 1u) == 0u) {
                payload[c * 16 + (lane >> 1)] = uchar(q0 | (q1 << 4));
            }
        } else if (bits == 6) {
            uint q0 = q & 0x3Fu;
            uint q1 = simd_shuffle_xor(q0, 1);
            uint q2 = simd_shuffle_xor(q0, 2);
            uint q3 = simd_shuffle_xor(q0, 3);
            if ((lane & 3u) == 0u) {
                int off = c * 24 + int(lane >> 2) * 3;
                payload[off + 0] = uchar(q0 | ((q1 & 0x3u) << 6));
                payload[off + 1] = uchar((q1 >> 2) | ((q2 & 0xFu) << 4));
                payload[off + 2] = uchar((q2 >> 4) | (q3 << 2));
            }
        } else if (bits == 8) {
            payload[c * 32 + int(lane)] = uchar(q & 0xFFu);
        }
    }
}
"""


# Compile and cache the fused Metal SIMD HQ quantization kernel when MPS is available.
def _get_mps_quant_lib():
    global _MPS_QUANT_LIB, _MPS_QUANT_FAILED
    if _MPS_QUANT_LIB is not None or _MPS_QUANT_FAILED:
        return _MPS_QUANT_LIB
    try:
        import torch
        if torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"):
            _MPS_QUANT_LIB = torch.mps.compile_shader(_MPS_QUANT_SRC)
        else:
            _MPS_QUANT_FAILED = True
    except Exception:
        _MPS_QUANT_FAILED = True
    return _MPS_QUANT_LIB


# Find per-group MSE-optimal (scale_f16, bias_f16) over a shrink grid for low-bit (2/3-bit) quantization.
def _mse_optimal_scale_bias(wf, w_min, w_max, qmax: float):
    import torch
    mid = 0.5 * (w_max + w_min)
    half_span = 0.5 * (w_max - w_min)
    best_scale = ((w_max - w_min) / qmax).to(torch.float16)
    best_bias = w_min.to(torch.float16)
    best_err = torch.full(w_min.shape, float("inf"), device=wf.device, dtype=torch.float32)
    for shrink in (1.00, 0.96, 0.92, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70):
        c_min = mid - shrink * half_span
        c_max = mid + shrink * half_span
        sc16 = ((c_max - c_min) / qmax).to(torch.float16)
        bi16 = c_min.to(torch.float16)
        sc32 = sc16.to(torch.float32).unsqueeze(-1)
        bi32 = bi16.to(torch.float32).unsqueeze(-1)
        safe_sc = torch.where(sc32 > 0.0, sc32, torch.ones_like(sc32))
        q = torch.round((wf - bi32) / safe_sc).clamp(0.0, qmax)
        q = torch.where(sc32 > 0.0, q, torch.zeros_like(q))
        recon = q * sc32 + bi32
        err = ((wf - recon) ** 2).mean(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, sc16, best_scale)
        best_bias = torch.where(better, bi16, best_bias)
    return best_scale, best_bias


# Quantize 2D tensor [N, K] into (blocks [N, K // eff_group, block_bytes], eff_group).
def quant_hq_blocks(
    w,
    bits: int = 4,
    group: int = 32,
    is_m5: bool = False,
    rotate: bool = True,
    mse_clip: bool | None = None,
) -> tuple[np.ndarray, int]:
    import torch
    if not isinstance(w, torch.Tensor):
        w = torch.as_tensor(np.asarray(w))
    if w.ndim != 2:
        raise ValueError(f"quant_hq_blocks expects 2D tensor, got {tuple(w.shape)}")
    n, k = w.shape
    eff_group = resolve_group_size(k, group)
    if bits == 4 and eff_group == 32 and is_m5:
        if n > 0 and k > 0 and bool(torch.isfinite(w).all()):
            mps_lib = _get_mps_quant_lib()
            total_groups = n * (k // 32)
            if mps_lib is not None and total_groups <= (1 << 24):
                w_mps = w.to(device="mps", dtype=torch.float32).contiguous()
                out_mps = torch.empty((n, k // 32, 20), dtype=torch.uint8, device="mps")
                mps_lib.quant_hq4_m5_simd(
                    w_mps, out_mps, total_groups, 1 if rotate else 0,
                    threads=(total_groups * 32, 1, 1), group_size=(256, 1, 1),
                )
                return out_mps.cpu().numpy(), 32
        return quant_hq4_blocks(w, rotate=rotate), 32
    if bits not in VALID_BITS:
        raise ValueError(f"Unsupported bits={bits}; expected {VALID_BITS}")
    if n == 0 or k == 0:
        return np.zeros((n, k // eff_group, hq_block_bytes(bits, eff_group)), dtype=np.uint8), eff_group
    if not bool(torch.isfinite(w).all()):
        raise ValueError("quant_hq_blocks rejects non-finite weights (NaN/Inf)")

    use_mse_clip = (bits in (2, 3)) if mse_clip is None else bool(mse_clip)
    nb = k // eff_group
    bbytes = hq_block_bytes(bits, eff_group)
    mps_lib = _get_mps_quant_lib()
    if mps_lib is not None and (n * nb) <= (1 << 24):
        w_mps = w.to(device="mps", dtype=torch.float32).contiguous()
        total_groups = n * nb
        out_mps = torch.empty((n, nb, bbytes), dtype=torch.uint8, device="mps")
        mps_lib.quant_hq_simd(
            w_mps, out_mps, total_groups, int(bits), int(eff_group),
            1 if rotate else 0, 1 if use_mse_clip else 0,
            threads=(total_groups * 32, 1, 1), group_size=(256, 1, 1),
        )
        return out_mps.cpu().numpy(), eff_group

    device = w.device if w.device.type in ("mps", "cuda") else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    qmax = float((1 << bits) - 1)
    out = np.empty((n, nb, bbytes), dtype=np.uint8)
    step = 4096 if use_mse_clip else 8192
    with torch.no_grad():
        for r0 in range(0, n, step):
            r1 = min(r0 + step, n)
            rn = r1 - r0
            wf = w[r0:r1].to(device=device, dtype=torch.float32)
            if rotate:
                wf = _fwht32_torch(wf.reshape(rn, k // GROUP, GROUP)).reshape(rn, nb, eff_group)
            else:
                wf = wf.reshape(rn, nb, eff_group)
            w_min = wf.amin(dim=-1)
            w_max = wf.amax(dim=-1)
            if use_mse_clip:
                scale_f16, bias_f16 = _mse_optimal_scale_bias(wf, w_min, w_max, qmax)
            else:
                scale_f16 = ((w_max - w_min) / qmax).to(torch.float16)
                bias_f16 = w_min.to(torch.float16)
            scale_r = scale_f16.to(torch.float32).unsqueeze(-1)
            bias_r = bias_f16.to(torch.float32).unsqueeze(-1)
            safe_scale = torch.where(scale_r > 0.0, scale_r, torch.ones_like(scale_r))
            q = torch.round((wf - bias_r) / safe_scale).clamp(0.0, qmax).to(torch.uint8)
            q = torch.where(scale_r > 0.0, q, torch.zeros_like(q))
            payload = _pack_bitstream(q, bits)
            header = torch.stack([scale_f16.view(torch.uint16), bias_f16.view(torch.uint16)], dim=-1).view(torch.uint8)
            blk = torch.cat([header, payload], dim=-1)
            out[r0:r1] = blk.cpu().numpy()
    return out, eff_group


# Dequantize HQ blocks to rotated-domain float32 [N, K].
def dequant_hq_blocks_rot(
    blocks: np.ndarray,
    bits: int = 4,
    group: int = 32,
    is_m5: bool = False,
    f16_round: bool = False,
) -> np.ndarray:
    if bits == 4 and group == 32 and is_m5:
        return dequant_hq4_g32_m5_rot(blocks, f16_round=f16_round)
    b = np.ascontiguousarray(blocks, dtype=np.uint8)
    bbytes = hq_block_bytes(bits, group)
    if b.ndim != 3 or b.shape[2] != bbytes:
        raise ValueError(f"Expected (N, nb, {bbytes}) blocks, got {b.shape}")
    n, nb, _ = b.shape
    scales = b[:, :, 0:2].copy().view(np.float16)[:, :, 0].astype(np.float32)[:, :, None]
    biases = b[:, :, 2:4].copy().view(np.float16)[:, :, 0].astype(np.float32)[:, :, None]
    q = _unpack_bitstream(b[:, :, 4:], bits, group)
    w_rot = q * scales + biases
    if f16_round:
        w_rot = w_rot.astype(np.float16).astype(np.float32)
    return w_rot.reshape(n, nb * group)


# Dequantize HQ blocks back to original weight domain float32 [N, K].
def dequant_hq_blocks(
    blocks: np.ndarray,
    bits: int = 4,
    group: int = 32,
    is_m5: bool = False,
    f16_round: bool = False,
) -> np.ndarray:
    if bits == 4 and group == 32 and is_m5 and not f16_round:
        return dequant_hq4_g32_m5(blocks)
    w_rot = dequant_hq_blocks_rot(blocks, bits=bits, group=group, is_m5=is_m5, f16_round=f16_round)
    n, k = w_rot.shape
    if n == 0 or k == 0:
        return w_rot
    import torch
    with torch.no_grad():
        t = torch.from_numpy(w_rot).reshape(n, k // GROUP, GROUP)
        return _fwht32_torch(t).reshape(n, k).numpy()


def quant_hq8_g32_m5(w):
    """Master BF16/FP32 -> symmetric Hadamard-rotated INT8 planar planes."""
    import torch
    if isinstance(w, torch.Tensor):
        if w.ndim != 2 or w.shape[0] <= 0 or w.shape[1] <= 0 or w.shape[1] % GROUP:
            raise ValueError(f"HQ8_G32_M5 requires a non-empty 2-D tensor with K divisible by {GROUP}")
        if not torch.isfinite(w).all():
            raise ValueError("HQ8_G32_M5 source weights must be finite")
        w_t = w
    else:
        w_arr = np.asarray(w, dtype=np.float32)
        if w_arr.ndim != 2 or w_arr.shape[0] <= 0 or w_arr.shape[1] <= 0 or w_arr.shape[1] % GROUP:
            raise ValueError(f"HQ8_G32_M5 requires a non-empty 2-D tensor with K divisible by {GROUP}")
        if not np.isfinite(w_arr).all():
            raise ValueError("HQ8_G32_M5 source weights must be finite")
        w_t = torch.from_numpy(np.ascontiguousarray(w_arr))
    N, K = w_t.shape
    G = K // GROUP
    inv_sqrt32 = float(INV_SQRT32)
    qs = np.empty((N, K), dtype=np.int8)
    scales = np.empty((N, G), dtype=np.float16)
    step = 8192
    for r0 in range(0, N, step):
        r1 = min(r0 + step, N)
        rn = r1 - r0
        t = w_t[r0:r1].float() if w_t.dtype != torch.float32 else w_t[r0:r1]
        h = 1
        while h < 32:
            t = t.reshape(-1, 32 // (2 * h), 2, h)
            u = t[:, :, 0, :]
            v = t[:, :, 1, :]
            t = torch.cat((u + v, u - v), dim=-1)
            h *= 2
        w_rot = t.reshape(rn, G, GROUP) * inv_sqrt32
        amax = w_rot.abs().amax(dim=-1)
        scale_f16 = (amax / 127.0).to(torch.float16)
        exact = scale_f16.float().unsqueeze(-1)
        safe = torch.where(exact > 0, exact, torch.ones_like(exact))
        q = torch.where(exact > 0,
                        torch.clamp(torch.round(w_rot / safe), -127.0, 127.0),
                        torch.zeros_like(w_rot))
        qs[r0:r1] = q.reshape(rn, K).to(torch.int8).numpy()
        scales[r0:r1] = scale_f16.numpy()
    return qs, scales


def dequant_hq8_g32_m5_rot(qs, scales):
    qs = np.asarray(qs)
    sc = np.asarray(scales, dtype=np.float16).astype(np.float32)
    N, K = qs.shape
    return (qs.astype(np.float32).reshape(N, K // GROUP, GROUP)
            * sc[:, :, None]).reshape(N, K)


def dequant_hq8_g32_m5(qs, scales):
    from hq4 import fwht32_ortho
    return fwht32_ortho(dequant_hq8_g32_m5_rot(qs, scales))

