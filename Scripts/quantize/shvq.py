# Signed Hadamard e-means Vector Quantizer (SHVQ{2,3,4}_G{32,64,128,256}).
# Factors exact coordinate signs + positive-orthant e-means codebooks with 2B f16
# group scale headers (zero bias) for 0-ULP Python/Swift/Metal parity.
from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
from pathlib import Path

import numpy as np

from emeans import EmeansRegime, fit_emeans
from hq4 import BLOCK_BYTES, INV_SQRT32, fwht32_ortho
from hq_blocks import (
    dequant_hq_blocks,
    format_quant_spec,
    hq_block_bytes,
    parse_quant_spec,
    quant_hq_blocks,
)

SHVQ_VALID_BITS = (2, 3, 4)
SHVQ_VALID_GROUPS = (32, 64, 128, 256)
SHVQ_PREFIXES = ("shvq2", "shvq3", "shvq4")
SHVQ_TABLE_MAGIC = b"QSHVQT01"
TABLES_BIN_PATH = Path(__file__).resolve().parent / "shvq_tables.bin"
METAL_ENC_SRC_PATH = Path(__file__).resolve().parent / "shvq_metal_enc.mm"

_CACHED_TABLES: dict[str, np.ndarray] | None = None
_METAL_ENC_LIB = None
_METAL_ENC_FAILED = False


def _get_metal_enc_lib():
    global _METAL_ENC_LIB, _METAL_ENC_FAILED
    if _METAL_ENC_LIB is not None or _METAL_ENC_FAILED:
        return _METAL_ENC_LIB
    import ctypes
    import platform
    import subprocess
    import tempfile

    if platform.system() != "Darwin" or not METAL_ENC_SRC_PATH.is_file():
        _METAL_ENC_FAILED = True
        return None
    try:
        src_sha = hashlib.sha256(METAL_ENC_SRC_PATH.read_bytes()).hexdigest()[:16]
        dylib_path = Path(tempfile.gettempdir()) / f"qwg_shvq_metal_enc_{src_sha}.dylib"
        if not dylib_path.is_file():
            tmp_out = dylib_path.with_suffix(f".{os.getpid()}.tmp.dylib")
            subprocess.run(
                [
                    "clang++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ObjC++",
                    str(METAL_ENC_SRC_PATH), "-framework", "Metal", "-framework", "Foundation",
                    "-o", str(tmp_out),
                ],
                check=True,
                capture_output=True,
            )
            tmp_out.replace(dylib_path)
        lib = ctypes.CDLL(str(dylib_path))
        lib.shvq_metal_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.shvq_metal_init.restype = ctypes.c_int
        lib.shvq_metal_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        lib.shvq_metal_encode.restype = ctypes.c_int
        tables = load_shvq_tables()
        cb2 = np.ascontiguousarray(tables["lut2_bits"].view(np.float16).astype(np.float32))
        cb3 = np.ascontiguousarray(tables["lut3_bits"].view(np.float16).astype(np.float32))
        smap = tables["step_map_bits"].view(np.float16).astype(np.float32)
        cb4 = np.ascontiguousarray(smap[tables["lut4_u8"]], dtype=np.float32)
        if lib.shvq_metal_init(cb2.ctypes.data, cb3.ctypes.data, cb4.ctypes.data) != 0:
            _METAL_ENC_FAILED = True
            return None
        _METAL_ENC_LIB = lib
    except Exception:
        _METAL_ENC_FAILED = True
    return _METAL_ENC_LIB


def is_shvq_spec(spec: str) -> bool:
    s = spec.strip().lower()
    return any(s == p or s.startswith(p + "_g") for p in SHVQ_PREFIXES)


def parse_shvq_spec(spec: str, default_group: int | None = None) -> tuple[int, int]:
    s = spec.strip().lower()
    if default_group is not None and default_group not in SHVQ_VALID_GROUPS:
        raise ValueError(f"Unsupported SHVQ group {default_group}; expected {SHVQ_VALID_GROUPS}")
    if s in SHVQ_PREFIXES:
        return int(s[4:]), (default_group if default_group is not None else 64)
    parts = s.split("_")
    if len(parts) == 2 and parts[0] in SHVQ_PREFIXES and parts[1].startswith("g"):
        b = int(parts[0][4:])
        g = int(parts[1][1:])
        if b in SHVQ_VALID_BITS and g in SHVQ_VALID_GROUPS:
            return b, g
    raise ValueError(f"Invalid SHVQ spec {spec!r}; expected shvq{{2,3,4}}_g{{32,64,128,256}}")


def format_shvq_spec(bits: int, group: int) -> str:
    if bits not in SHVQ_VALID_BITS or group not in SHVQ_VALID_GROUPS:
        raise ValueError(f"Unsupported SHVQ (bits={bits}, group={group})")
    return f"shvq{bits}_g{group}"


def shvq_block_bytes(bits: int, group: int) -> int:
    if bits not in SHVQ_VALID_BITS or group not in SHVQ_VALID_GROUPS:
        raise ValueError(f"Unsupported SHVQ (bits={bits}, group={group})")
    return 2 + (group * bits) // 8


def parse_any_quant_spec(spec: str, default_group: int | None = None) -> tuple[int, int, bool, bool]:
    if is_shvq_spec(spec):
        b, g = parse_shvq_spec(spec, default_group=default_group)
        return b, g, False, True
    b, g, is_m5 = parse_quant_spec(spec, default_group=default_group)
    return b, g, is_m5, False


def format_any_quant_spec(bits: int, group: int, is_m5: bool = False, is_shvq: bool = False) -> str:
    if is_shvq:
        return format_shvq_spec(bits, group)
    return format_quant_spec(bits, group, is_m5=is_m5)


def any_block_bytes(bits: int, group: int, is_m5: bool = False, is_shvq: bool = False) -> int:
    if is_shvq:
        return shvq_block_bytes(bits, group)
    if bits == 4 and group == 32 and is_m5:
        return BLOCK_BYTES
    return hq_block_bytes(bits, group)


def quant_any_blocks(w, bits: int, group: int, is_m5: bool = False, is_shvq: bool = False):
    if is_shvq:
        return quant_shvq_blocks(w, bits=bits, group=group)
    return quant_hq_blocks(w, bits=bits, group=group, is_m5=is_m5)


def dequant_any_blocks(blocks: np.ndarray, bits: int, group: int, is_m5: bool = False, is_shvq: bool = False) -> np.ndarray:
    if is_shvq:
        return dequant_shvq_blocks(blocks, bits=bits, group=group)
    return dequant_hq_blocks(blocks, bits=bits, group=group, is_m5=is_m5)


def fit_canonical_shvq_tables(device: str = "cpu") -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261110)
    g64 = rng.standard_normal((32768, 64), dtype=np.float32)
    g64 /= np.sqrt(np.mean(g64 * g64, axis=1, keepdims=True)).clip(1e-8)
    abs_g = np.abs(g64)

    c2 = fit_emeans(
        abs_g.reshape(-1, 8), k=256, regime=EmeansRegime.IN_SAMPLE_STATIC,
        epochs=2.5, seed=42, device=device, polish_steps=25
    ).centroids
    c2 = np.clip(c2, 0.0, 65504.0).astype(np.float16)
    order2 = np.argsort(np.linalg.norm(c2.astype(np.float32), axis=1))
    lut2_bits = c2[order2].view(np.uint16).copy()

    c3 = fit_emeans(
        abs_g.reshape(-1, 4), k=256, regime=EmeansRegime.IN_SAMPLE_STATIC,
        epochs=2.5, seed=43, device=device, polish_steps=25
    ).centroids
    c3 = np.clip(c3, 0.0, 65504.0).astype(np.float16)
    order3 = np.argsort(np.linalg.norm(c3.astype(np.float32), axis=1))
    lut3_bits = c3[order3].view(np.uint16).copy()

    c4 = fit_emeans(
        abs_g.reshape(-1, 4), k=4096, regime=EmeansRegime.IN_SAMPLE_STATIC,
        epochs=2.5, seed=44, device=device, polish_steps=25
    ).centroids
    c4 = np.clip(c4, 0.0, 65504.0)
    order4 = np.argsort(np.linalg.norm(c4, axis=1))
    c4 = c4[order4]
    lut4_max = float(np.float16(c4.max()))
    step_map_f16 = (np.arange(256, dtype=np.float32) * (lut4_max / 255.0)).astype(np.float16)
    step_map_bits = step_map_f16.view(np.uint16).copy()
    lut4_u8 = np.rint(np.clip(c4 / lut4_max * 255.0, 0.0, 255.0)).astype(np.uint8)

    return {
        "lut2_bits": lut2_bits,
        "lut3_bits": lut3_bits,
        "lut4_u8": lut4_u8,
        "step_map_bits": step_map_bits,
    }


def save_tables_bin(tables: dict[str, np.ndarray], path: Path = TABLES_BIN_PATH) -> str:
    payload = (
        SHVQ_TABLE_MAGIC
        + tables["lut2_bits"].astype("<u2").tobytes()
        + tables["lut3_bits"].astype("<u2").tobytes()
        + tables["step_map_bits"].astype("<u2").tobytes()
        + tables["lut4_u8"].astype(np.uint8).tobytes()
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_shvq_tables() -> dict[str, np.ndarray]:
    global _CACHED_TABLES
    if _CACHED_TABLES is not None:
        return _CACHED_TABLES
    if not TABLES_BIN_PATH.is_file():
        tables = fit_canonical_shvq_tables(device="cpu")
        save_tables_bin(tables, TABLES_BIN_PATH)
        _CACHED_TABLES = tables
        return tables
    raw = TABLES_BIN_PATH.read_bytes()
    expected = 8 + (256 * 8 * 2) + (256 * 4 * 2) + (256 * 2) + (4096 * 4)
    if len(raw) != expected or raw[:8] != SHVQ_TABLE_MAGIC:
        raise ValueError(f"Corrupt {TABLES_BIN_PATH}: len={len(raw)} expected={expected}")
    off = 8
    lut2_bits = np.frombuffer(raw, dtype="<u2", count=256 * 8, offset=off).reshape(256, 8).copy()
    off += 256 * 8 * 2
    lut3_bits = np.frombuffer(raw, dtype="<u2", count=256 * 4, offset=off).reshape(256, 4).copy()
    off += 256 * 4 * 2
    step_map_bits = np.frombuffer(raw, dtype="<u2", count=256, offset=off).copy()
    off += 256 * 2
    lut4_u8 = np.frombuffer(raw, dtype=np.uint8, count=4096 * 4, offset=off).reshape(4096, 4).copy()
    _CACHED_TABLES = {
        "lut2_bits": lut2_bits,
        "lut3_bits": lut3_bits,
        "lut4_u8": lut4_u8,
        "step_map_bits": step_map_bits,
    }
    return _CACHED_TABLES


def _fwht32_torch(t):
    import torch
    orig = t.shape
    h = 1
    while h < 32:
        t = t.reshape(-1, 32 // (2 * h), 2, h)
        t = torch.cat((t[:, :, 0, :] + t[:, :, 1, :], t[:, :, 0, :] - t[:, :, 1, :]), dim=-1)
        h *= 2
    return t.reshape(orig) * float(INV_SQRT32)


def quant_shvq_blocks(
    w,
    bits: int,
    group: int,
    rotate: bool = True,
    als_iters: int = 2,
    device: str | None = None,
) -> tuple[np.ndarray, int]:
    import torch
    if bits not in SHVQ_VALID_BITS:
        raise ValueError(f"Unsupported SHVQ bits={bits}; expected {SHVQ_VALID_BITS}")
    if group not in SHVQ_VALID_GROUPS:
        raise ValueError(f"Unsupported SHVQ group={group}; expected {SHVQ_VALID_GROUPS}")
    wt = w.detach() if isinstance(w, torch.Tensor) else torch.as_tensor(w)
    if wt.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix [N, K], got {tuple(wt.shape)}")
    n_rows, k_cols = wt.shape
    eff_group = math.gcd(k_cols, group)
    if eff_group not in SHVQ_VALID_GROUPS:
        raise ValueError(f"K={k_cols} incompatible with SHVQ group={group}")
    if device is None:
        dev = wt.device if wt.device.type == "mps" else (
            torch.device("mps") if torch.backends.mps.is_available() and wt.numel() >= 65536 else torch.device("cpu")
        )
    else:
        dev = torch.device(device)

    tables = load_shvq_tables()
    bbytes = shvq_block_bytes(bits, eff_group)
    n_groups = k_cols // eff_group
    out = np.empty((n_rows, n_groups, bbytes), dtype=np.uint8)

    # Native Metal register encoder (fixed 146 MiB shared buffer, zero PyTorch MPS overspill).
    if device is None or device == "mps":
        mlib = _get_metal_enc_lib()
        if mlib is not None and n_rows > 0:
            total_groups = n_rows * n_groups
            if isinstance(wt, torch.Tensor) and wt.device.type == "cpu" and wt.dtype == torch.bfloat16:
                raw_u16 = np.ascontiguousarray(wt.view(torch.uint16).numpy())
                if bool(torch.isfinite(wt[: min(n_rows, 16)]).all()):
                    rc = mlib.shvq_metal_encode(
                        raw_u16.ctypes.data, out.ctypes.data, total_groups,
                        eff_group, bits, 1 if rotate else 0, 1, als_iters,
                    )
                    if rc == 0:
                        return out, eff_group
            else:
                w_cpu = np.ascontiguousarray(
                    wt.detach().cpu().to(torch.float32).numpy()
                    if isinstance(wt, torch.Tensor)
                    else np.asarray(w, dtype=np.float32)
                )
                if np.isfinite(w_cpu[: min(n_rows, 16)]).all():
                    rc = mlib.shvq_metal_encode(
                        w_cpu.ctypes.data, out.ctypes.data, total_groups,
                        eff_group, bits, 1 if rotate else 0, 0, als_iters,
                    )
                    if rc == 0:
                        return out, eff_group

    if bits == 2:
        cb_f32 = tables["lut2_bits"].view(np.float16).astype(np.float32)
        sub_d = 8
    elif bits == 3:
        cb_f32 = tables["lut3_bits"].view(np.float16).astype(np.float32)
        sub_d = 4
    else:
        smap = tables["step_map_bits"].view(np.float16).astype(np.float32)
        cb_f32 = smap[tables["lut4_u8"]]
        sub_d = 4

    cb = torch.from_numpy(cb_f32).to(device=dev, dtype=torch.float32)
    cb_n2 = (cb * cb).sum(dim=-1)

    row_chunk = max(1, min(n_rows, (32 * 1024 * 1024) // max(1, k_cols)))
    bs_sub = 131072 if bits == 4 else 524288
    bit_weights8 = (1 << np.arange(8, dtype=np.uint8))[None, None, :]
    bit_weights4 = (1 << np.arange(4, dtype=np.uint16))[None, None, :]

    with torch.no_grad():
        for r0 in range(0, n_rows, row_chunk):
            r1 = min(n_rows, r0 + row_chunk)
            rc = r1 - r0
            wf = wt[r0:r1].to(device=dev, dtype=torch.float32)
            if rotate:
                wf = _fwht32_torch(wf.reshape(-1, 32)).reshape(rc, k_cols)
            wg = wf.reshape(rc * n_groups, eff_group)
            neg = torch.signbit(wg)
            w_abs = wg.abs()
            scale = torch.sqrt(torch.mean(w_abs * w_abs, dim=-1, keepdim=True)).clamp(min=1e-8, max=65504.0)
            idx_all = None
            for it in range(als_iters + 1):
                sc16 = scale.to(torch.float16).to(torch.float32).clamp(min=1e-8)
                u = (w_abs / sc16).reshape(-1, sub_d)
                idx_u = torch.empty(u.shape[0], device=dev, dtype=torch.int32)
                for i in range(0, u.shape[0], bs_sub):
                    ub = u[i : i + bs_sub]
                    idx_u[i : i + bs_sub] = (cb_n2[None, :] - 2.0 * (ub @ cb.t())).argmin(dim=-1).to(torch.int32)
                if it < als_iters:
                    c_mat = cb[idx_u].reshape(-1, eff_group)
                    num = (w_abs * c_mat).sum(dim=-1, keepdim=True)
                    den = (c_mat * c_mat).sum(dim=-1, keepdim=True).clamp(min=1e-12)
                    scale = (num / den).clamp(min=1e-8, max=65504.0)
                else:
                    idx_all = idx_u.cpu().numpy()

            sc_u16 = scale.squeeze(-1).to(torch.float16).cpu().numpy().view(np.uint16).reshape(rc, n_groups)
            out[r0:r1, :, 0] = (sc_u16 & 0xFF).astype(np.uint8)
            out[r0:r1, :, 1] = (sc_u16 >> 8).astype(np.uint8)
            neg_np = neg.cpu().numpy().astype(np.uint8)
            n_oct = eff_group // 8

            if bits == 2:
                signs8 = (neg_np.reshape(rc * n_groups, n_oct, 8) * bit_weights8).sum(axis=-1).astype(np.uint8)
                idx8 = idx_all.reshape(rc * n_groups, n_oct).astype(np.uint8)
                pay = np.stack([signs8, idx8], axis=-1).reshape(rc, n_groups, n_oct * 2)
                out[r0:r1, :, 2:] = pay
            elif bits == 3:
                signs8 = (neg_np.reshape(rc * n_groups, n_oct, 8) * bit_weights8).sum(axis=-1).astype(np.uint8)
                idx_pair = idx_all.reshape(rc * n_groups, n_oct, 2).astype(np.uint8)
                pay = np.stack([signs8, idx_pair[:, :, 0], idx_pair[:, :, 1]], axis=-1).reshape(rc, n_groups, n_oct * 3)
                out[r0:r1, :, 2:] = pay
            else:
                signs4 = (neg_np.astype(np.uint16).reshape(rc * n_groups, n_oct * 2, 4) * bit_weights4).sum(axis=-1).astype(np.uint16)
                idx12 = (idx_all.reshape(rc * n_groups, n_oct * 2).astype(np.uint16) & 0x0FFF)
                u16_words = idx12 | (signs4 << 12)
                out[r0:r1, :, 2:] = u16_words.astype("<u2").view(np.uint8).reshape(rc, n_groups, n_oct * 4)

    return out, eff_group


def dequant_shvq_blocks_rot(blocks: np.ndarray, bits: int, group: int) -> np.ndarray:
    b = np.ascontiguousarray(blocks, dtype=np.uint8)
    n_rows, n_groups, bbytes = b.shape
    expected = shvq_block_bytes(bits, group)
    if bbytes != expected:
        raise ValueError(f"Expected {expected} bytes per block for shvq{bits}_g{group}, got {bbytes}")
    tables = load_shvq_tables()
    sc_bits = b[:, :, 0].astype(np.uint16) | (b[:, :, 1].astype(np.uint16) << 8)
    sc_f32 = sc_bits.view(np.float16).astype(np.float32)[:, :, None]
    pay = b[:, :, 2:]
    n_oct = group // 8

    if bits == 2:
        rec = pay.reshape(n_rows, n_groups, n_oct, 2)
        signs8 = rec[:, :, :, 0]
        idx8 = rec[:, :, :, 1]
        pos_bits = tables["lut2_bits"][idx8]
        s_bits = ((signs8[:, :, :, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint16) << 15
        w_bits = (pos_bits | s_bits).reshape(n_rows, n_groups, group)
    elif bits == 3:
        rec = pay.reshape(n_rows, n_groups, n_oct, 3)
        signs8 = rec[:, :, :, 0]
        idx0 = rec[:, :, :, 1]
        idx1 = rec[:, :, :, 2]
        pos0 = tables["lut3_bits"][idx0]
        pos1 = tables["lut3_bits"][idx1]
        pos_bits = np.concatenate([pos0, pos1], axis=-1)
        s_bits = ((signs8[:, :, :, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint16) << 15
        w_bits = (pos_bits | s_bits).reshape(n_rows, n_groups, group)
    else:
        u16 = pay.view("<u2").reshape(n_rows, n_groups, n_oct * 2)
        idx12 = u16 & 0x0FFF
        signs4 = (u16 >> 12) & 0x0F
        u8_codes = tables["lut4_u8"][idx12]
        pos_bits = tables["step_map_bits"][u8_codes]
        s_bits = ((signs4[:, :, :, None] >> np.arange(4, dtype=np.uint16)) & 1).astype(np.uint16) << 15
        w_bits = (pos_bits | s_bits).reshape(n_rows, n_groups, group)

    vals_f32 = w_bits.view(np.float16).astype(np.float32) * sc_f32
    return vals_f32.reshape(n_rows, n_groups * group)


def dequant_shvq_blocks(blocks: np.ndarray, bits: int, group: int, rotate: bool = True) -> np.ndarray:
    rot = dequant_shvq_blocks_rot(blocks, bits, group)
    if not rotate:
        return rot
    n_rows, k_cols = rot.shape
    return fwht32_ortho(rot.reshape(-1, 32)).reshape(n_rows, k_cols)


def emit_metal_and_swift_tables(metal_path: Path, swift_path: Path) -> None:
    tables = load_shvq_tables()
    l2 = tables["lut2_bits"].reshape(-1)
    l3 = tables["lut3_bits"].reshape(-1)
    sm = tables["step_map_bits"].reshape(-1)
    l4_u32 = tables["lut4_u8"].view("<u4").reshape(-1)

    def fmt_hex(arr: np.ndarray, width: int, suffix: str = "u", per_line: int = 16) -> list[str]:
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = ", ".join(f"0x{int(v):0{width}X}{suffix}" for v in arr[i : i + per_line])
            lines.append("    " + chunk + ("," if i + per_line < len(arr) else ""))
        return lines

    step_scale_f32 = np.float32(float(sm[255].view(np.float16)) / 255.0)
    m_lines = [
        "// qw_shvq_tables.metalh — Canonical e-means (arXiv:2611.09501) positive-orthant",
        "// codebooks for SHVQ{2,3,4}_G{32,64,128,256}. Generated by Scripts/quantize/shvq.py.",
        "#pragma once",
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        f"constant float kShvq4StepScaleF32 = {float(step_scale_f32).hex()}f;",
        "",
        "alignas(16) constant ushort kShvqLut2Bits[2048] = {",
        *fmt_hex(l2, 4),
        "};",
        "",
        "alignas(16) constant ushort kShvqLut3Bits[1024] = {",
        *fmt_hex(l3, 4),
        "};",
        "",
        "alignas(16) constant uint kShvqLut4PackedU8[4096] = {",
        *fmt_hex(l4_u32, 8),
        "};",
        "",
    ]
    metal_path.write_text("\n".join(m_lines), encoding="utf-8")

    direct_bits = sm[tables["lut4_u8"]].reshape(-1)
    routed_lines = [
        "// qw_shvq_routed_lut.metalh — Exact half-value SHVQ4 routed lookup.",
        "// Generated by Scripts/quantize/shvq.py; dense kernels retain the packed table.",
        "#pragma once",
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        "alignas(16) constant ushort kShvqLut4DirectBits[16384] = {",
        *fmt_hex(direct_bits, 4, per_line=32),
        "};",
        "",
    ]
    metal_path.with_name("qw_shvq_routed_lut.metalh").write_text(
        "\n".join(routed_lines), encoding="utf-8")

    s_lines = [
        "// QWSHVQTables.swift — Canonical e-means (arXiv:2611.09501) positive-orthant",
        "// codebooks for SHVQ{2,3,4}_G{32,64,128,256}. Generated by Scripts/quantize/shvq.py.",
        "",
        "enum QWSHVQTables {",
        "    static let lut2Bits: [UInt16] = [",
        *[("    " + ln) for ln in fmt_hex(l2, 4, suffix="")],
        "    ]",
        "",
        "    static let lut3Bits: [UInt16] = [",
        *[("    " + ln) for ln in fmt_hex(l3, 4, suffix="")],
        "    ]",
        "",
        "    static let stepMap4Bits: [UInt16] = [",
        *[("    " + ln) for ln in fmt_hex(sm, 4, suffix="")],
        "    ]",
        "",
        "    static let lut4PackedU8: [UInt32] = [",
        *[("    " + ln) for ln in fmt_hex(l4_u32, 8, suffix="")],
        "    ]",
        "}",
        "",
    ]
    swift_path.write_text("\n".join(s_lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-tables", action="store_true")
    args = ap.parse_args()
    if args.emit_tables:
        root = Path(__file__).resolve().parent.parent.parent
        tbls = fit_canonical_shvq_tables(device="cpu")
        digest = save_tables_bin(tbls, TABLES_BIN_PATH)
        emit_metal_and_swift_tables(
            root / "Sources" / "QWMetal" / "include" / "qw_shvq_tables.metalh",
            root / "Sources" / "Foundation" / "QWQuant" / "QWSHVQTables.swift",
        )
        print(f"emitted SHVQ tables sha256={digest}")
