# Antipodal Hadamard e-means sub-2-bit codebooks (SHVQ1 & SHVQ15).
# Factors 1 antipodal sign bit per 8D oct-vector + hemisphere e-means codebooks
# with exact dyadic int8 step mapping for 0-ULP Python/Swift/Metal parity.
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from emeans import EmeansRegime, fit_emeans

SHVQ_SUB2_MAGIC = b"QSHVQS01"
SUB2_TABLES_BIN_PATH = Path(__file__).resolve().parent / "shvq_sub2_tables.bin"
SHVQ15_STEP_SCALE_F32 = np.float32(3.0 / 128.0)
_F16_MIN_NORMAL = 6.103515625e-05

_CACHED_SUB2_TABLES: dict[str, np.ndarray] | None = None


def fit_antipodal_8d_codebook(
    samples: np.ndarray, k: int, iters: int = 30, seed: int = 104
) -> np.ndarray:
    import torch

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    n = x.shape[0]
    hemi = samples.astype(np.float32, copy=True)
    hemi[hemi[:, 0] < 0.0] *= -1.0
    init_cb = fit_emeans(
        hemi,
        k=k,
        regime=EmeansRegime.IN_SAMPLE_STATIC,
        epochs=2.5,
        seed=seed,
        device="cpu",
        polish_steps=20,
    ).centroids
    cb = torch.from_numpy(init_cb.astype(np.float32))
    bs = 131072
    with torch.no_grad():
        for _ in range(iters):
            cb_n2 = (cb * cb).sum(dim=-1)
            new_sum = torch.zeros_like(cb)
            counts = torch.zeros(k, dtype=torch.float32)
            for i in range(0, n, bs):
                xb = x[i : i + bs]
                dots = xb @ cb.t()
                dist = cb_n2[None, :] - 2.0 * dots.abs()
                idx = dist.argmin(dim=-1)
                sel_dots = dots.gather(1, idx[:, None]).squeeze(1)
                signs = torch.where(sel_dots >= 0.0, 1.0, -1.0)[:, None]
                new_sum.index_add_(0, idx, xb * signs)
                counts.index_add_(0, idx, torch.ones(xb.shape[0]))
            live = counts > 0
            cb[live] = new_sum[live] / counts[live, None]
            dead = ~live
            if bool(dead.any()):
                ridx = rng.integers(0, n, size=int(dead.sum()))
                cb[dead] = x[ridx]
    arr = cb.numpy()
    order = np.argsort(np.linalg.norm(arr, axis=1))
    return arr[order]


def fit_canonical_shvq_sub2_tables() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261111)
    g64 = rng.standard_normal((32768, 64), dtype=np.float32)
    g64 /= np.sqrt(np.mean(g64 * g64, axis=1, keepdims=True)).clip(1e-8)
    octs = g64.reshape(-1, 8)

    c1 = fit_antipodal_8d_codebook(octs, k=128, iters=40, seed=105)
    c1_nudge = np.where(
        np.abs(c1) < _F16_MIN_NORMAL,
        np.where(c1 >= 0.0, _F16_MIN_NORMAL, -_F16_MIN_NORMAL),
        c1,
    )
    lut1_f16 = np.clip(c1_nudge, -65504.0, 65504.0).astype(np.float16)
    lut1_bits = lut1_f16.view(np.uint16).copy()

    c15 = fit_antipodal_8d_codebook(octs, k=2048, iters=30, seed=104)
    c15_max = max(float(np.abs(c15).max()), 1e-8)
    target_max = float(127.0 * float(SHVQ15_STEP_SCALE_F32))
    c15_scaled = c15 * np.float32(target_max / c15_max)
    q_i8 = np.rint(
        np.clip(c15_scaled / float(SHVQ15_STEP_SCALE_F32), -127.0, 127.0)
    ).astype(np.int8)
    zero_mask = q_i8 == 0
    if np.any(zero_mask):
        q_i8 = np.where(zero_mask, np.where(c15_scaled >= 0.0, 1, -1), q_i8).astype(np.int8)
    lut15_u8 = q_i8.view(np.uint8).copy()

    i8_lut = np.arange(256, dtype=np.uint8).view(np.int8)
    step_map15_f16 = (i8_lut.astype(np.float32) * SHVQ15_STEP_SCALE_F32).astype(np.float16)
    step_map15_bits = step_map15_f16.view(np.uint16).copy()

    return {
        "lut1_bits": lut1_bits,
        "step_map15_bits": step_map15_bits,
        "lut15_u8": lut15_u8,
    }


def save_sub2_tables_bin(
    tables: dict[str, np.ndarray], path: Path = SUB2_TABLES_BIN_PATH
) -> str:
    payload = (
        SHVQ_SUB2_MAGIC
        + tables["lut1_bits"].astype("<u2").tobytes()
        + tables["step_map15_bits"].astype("<u2").tobytes()
        + tables["lut15_u8"].astype(np.uint8).tobytes()
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_shvq_sub2_tables() -> dict[str, np.ndarray]:
    global _CACHED_SUB2_TABLES
    if _CACHED_SUB2_TABLES is not None:
        return _CACHED_SUB2_TABLES
    if not SUB2_TABLES_BIN_PATH.is_file():
        tables = fit_canonical_shvq_sub2_tables()
        save_sub2_tables_bin(tables, SUB2_TABLES_BIN_PATH)
        _CACHED_SUB2_TABLES = tables
        return tables
    raw = SUB2_TABLES_BIN_PATH.read_bytes()
    expected = 8 + (128 * 8 * 2) + (256 * 2) + (2048 * 8)
    if len(raw) != expected or raw[:8] != SHVQ_SUB2_MAGIC:
        raise ValueError(f"Corrupt {SUB2_TABLES_BIN_PATH}: len={len(raw)} expected={expected}")
    off = 8
    lut1_bits = np.frombuffer(raw, dtype="<u2", count=128 * 8, offset=off).reshape(128, 8).copy()
    off += 128 * 8 * 2
    step_map15_bits = np.frombuffer(raw, dtype="<u2", count=256, offset=off).copy()
    off += 256 * 2
    lut15_u8 = np.frombuffer(raw, dtype=np.uint8, count=2048 * 8, offset=off).reshape(2048, 8).copy()
    _CACHED_SUB2_TABLES = {
        "lut1_bits": lut1_bits,
        "step_map15_bits": step_map15_bits,
        "lut15_u8": lut15_u8,
    }
    return _CACHED_SUB2_TABLES


def emit_sub2_metal_and_swift_tables(metal_path: Path, swift_path: Path) -> None:
    tables = load_shvq_sub2_tables()
    l1 = tables["lut1_bits"].reshape(-1)
    sm15 = tables["step_map15_bits"].reshape(-1)
    l15_u32 = tables["lut15_u8"].view("<u4").reshape(-1)

    def fmt_hex(arr: np.ndarray, width: int, suffix: str = "u", per_line: int = 16) -> list[str]:
        lines = []
        for i in range(0, len(arr), per_line):
            chunk = ", ".join(f"0x{int(v):0{width}X}{suffix}" for v in arr[i : i + per_line])
            lines.append("    " + chunk + ("," if i + per_line < len(arr) else ""))
        return lines

    m_lines = [
        "// qw_shvq_sub2_tables.metalh — Canonical antipodal e-means hemisphere",
        "// codebooks for SHVQ{1,15}_G{32,64,128,256}. Generated by shvq_sub2.py.",
        "#pragma once",
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        f"constant float kShvq15StepScaleF32 = {float(SHVQ15_STEP_SCALE_F32).hex()}f;",
        "",
        "alignas(16) constant ushort kShvqLut1Bits[1024] = {",
        *fmt_hex(l1, 4),
        "};",
        "",
        "alignas(16) constant uint kShvqLut15PackedI8[4096] = {",
        *fmt_hex(l15_u32, 8),
        "};",
        "",
    ]
    metal_path.write_text("\n".join(m_lines), encoding="utf-8")

    direct_bits = sm15[tables["lut15_u8"]].reshape(-1)
    routed_lines = [
        "// qw_shvq_sub2_routed_lut.metalh — Exact half-value SHVQ15 routed lookup.",
        "// Generated by Scripts/quantize/shvq_sub2.py; dense kernels use packed int8.",
        "#pragma once",
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        "alignas(16) constant ushort kShvqLut15DirectBits[16384] = {",
        *fmt_hex(direct_bits, 4, per_line=32),
        "};",
        "",
    ]
    metal_path.with_name("qw_shvq_sub2_routed_lut.metalh").write_text(
        "\n".join(routed_lines), encoding="utf-8"
    )

    s_lines = [
        "// QWSHVQSub2Tables.swift — Canonical antipodal e-means hemisphere",
        "// codebooks for SHVQ{1,15}_G{32,64,128,256}. Generated by shvq_sub2.py.",
        "",
        "enum QWSHVQSub2Tables {",
        "    static let lut1Bits: [UInt16] = [",
        *[("    " + ln) for ln in fmt_hex(l1, 4, suffix="")],
        "    ]",
        "",
        "    static let stepMap15Bits: [UInt16] = [",
        *[("    " + ln) for ln in fmt_hex(sm15, 4, suffix="")],
        "    ]",
        "",
        "    static let lut15PackedI8: [UInt32] = [",
        *[("    " + ln) for ln in fmt_hex(l15_u32, 8, suffix="")],
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
        tbls = fit_canonical_shvq_sub2_tables()
        digest = save_sub2_tables_bin(tbls, SUB2_TABLES_BIN_PATH)
        emit_sub2_metal_and_swift_tables(
            root / "Sources" / "QWMetal" / "include" / "qw_shvq_sub2_tables.metalh",
            root / "Sources" / "Foundation" / "QWQuant" / "QWSHVQSub2Tables.swift",
        )
        print(f"emitted SHVQ sub-2-bit tables sha256={digest}")
