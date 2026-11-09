#!/usr/bin/env python3
"""Estimate native HQ pack costs without reading or requantizing weight payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from hq_blocks import (
    VALID_BITS, VALID_GROUPS, format_quant_spec, hq_block_bytes, resolve_group_size,
)
from verify import planes, read_header


def routed_expert(name: str, entry: dict) -> bool:
    return "experts" in entry or (".experts." in name and "shared_expert" not in name)


def candidate_cost(header: dict, original_bytes: int, dense: tuple[int, int],
                   expert: tuple[int, int]) -> dict:
    old_payload = 0
    new_payload = 0
    raw_bytes = 0
    groups: dict[str, int] = {}
    counts = {"routed_parameters": 0, "quantized_trunk_parameters": 0,
              "raw_parameters": 0}
    for name, entry in header["tensors"].items():
        count = math.prod(entry["shape"])
        current_bytes = sum(length for _, length, _ in planes(entry))
        old_payload += current_bytes
        if "raw_off" in entry:
            raw_bytes += current_bytes
            counts["raw_parameters"] += count
            new_payload += current_bytes
            continue
        is_expert = routed_expert(name, entry)
        bits, requested_group = expert if is_expert else dense
        group = resolve_group_size(entry["shape"][-1], requested_group)
        spec = format_quant_spec(bits, group)
        groups[spec] = groups.get(spec, 0) + 1
        counts["routed_parameters" if is_expert else "quantized_trunk_parameters"] += count
        new_payload += count // group * hq_block_bytes(bits, group)
    return {
        "dense": format_quant_spec(*dense),
        "expert": format_quant_spec(*expert),
        "payload_bytes": new_payload,
        "raw_bytes": raw_bytes,
        "estimated_pack_bytes": original_bytes + new_payload - old_payload,
        "estimated_pack_gib": (original_bytes + new_payload - old_payload) / 2**30,
        "effective_tensor_formats": groups,
        "parameters": counts,
    }


def audit_source(root: Path, expected_checksum: str | None = None) -> dict:
    """Validate payload extents and, optionally, the packer's aggregate source checksum."""
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source index must contain a non-empty weight_map")
    root = root.resolve()
    missing = []
    total_bytes = 0
    for file in sorted(set(weight_map.values())):
        if not isinstance(file, str):
            raise ValueError("weight_map file names must be strings")
        path = root / file
        if Path(file).is_absolute() or ".." in Path(file).parts:
            raise ValueError(f"shard path escapes source directory: {file}")
        if not path.is_file():
            missing.append(file)
            continue
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {file}")
            size = int.from_bytes(prefix, "little")
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header length: {file}")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"truncated safetensors header: {file}")
            tensors = json.loads(raw)
        payload_bytes = path.stat().st_size - 8 - size
        widths = {
            "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "F8_E8M0": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
        }
        for name, shard in weight_map.items():
            if shard != file:
                continue
            entry = tensors.get(name)
            if not isinstance(entry, dict):
                raise ValueError(f"indexed tensor missing from {file}: {name}")
            offsets = entry.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= payload_bytes):
                raise ValueError(f"incomplete/invalid payload for {name} in {file}")
            shape = entry.get("shape")
            width = widths.get(entry.get("dtype"))
            if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                    or width is None or math.prod(shape) * width != offsets[1] - offsets[0]):
                raise ValueError(f"tensor shape/dtype disagrees with payload extent: {name} in {file}")
        total_bytes += path.stat().st_size
    report = {
        "status": "incomplete" if missing else "payload-extents-verified",
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "indexed_tensors": len(weight_map),
        "shards": len(set(weight_map.values())),
        "missing_shards": missing,
        "available_bytes": total_bytes,
        "checksums_verified": False,
    }
    if expected_checksum is not None:
        if (not isinstance(expected_checksum, str) or len(expected_checksum) != 64
                or any(c not in "0123456789abcdef" for c in expected_checksum)):
            raise ValueError("incumbent provenance must contain a hexadecimal SHA256 source checksum")
        if missing:
            raise ValueError("cannot verify source checksum while indexed shards are missing")
        files = sorted(root.glob("*.safetensors"))
        if {file.name for file in files} != set(weight_map.values()):
            raise ValueError("source shards must exactly match the index for pack.py checksum verification")
        digest = hashlib.sha256()
        for file in files:
            print(f"hashing {file.name}", flush=True)
            with file.open("rb") as stream:
                while chunk := stream.read(16 << 20):
                    digest.update(chunk)
        actual = digest.hexdigest()
        report.update({
            "source_sha256": actual,
            "expected_source_sha256": expected_checksum,
            "checksums_verified": actual == expected_checksum,
            "status": "source-checksum-verified" if actual == expected_checksum
                      else "source-checksum-mismatch",
        })
    return report


def pareto_frontier(measurements: list[dict]) -> list[dict]:
    """Compare measured bytes and teacher KL only within one reference/corpus."""
    if not measurements:
        return []
    required = {"candidate", "pack_bytes", "mean_kl", "teacher_sha256", "corpus_sha256"}
    for row in measurements:
        if not required <= row.keys():
            raise ValueError(f"measurement requires fields {sorted(required)}")
        if (type(row["pack_bytes"]) is not int or row["pack_bytes"] <= 0
                or type(row["mean_kl"]) not in (int, float)
                or not math.isfinite(row["mean_kl"]) or row["mean_kl"] < 0):
            raise ValueError("measurements require positive actual bytes and finite nonnegative KL")
        if any(not isinstance(row[key], str) or not row[key]
               for key in ("candidate", "teacher_sha256", "corpus_sha256")):
            raise ValueError("measurement identities must be non-empty strings")
    if len({row["candidate"] for row in measurements}) != len(measurements):
        raise ValueError("duplicate candidate measurement")
    if len({(row["teacher_sha256"], row["corpus_sha256"]) for row in measurements}) != 1:
        raise ValueError("cannot compare different teachers or scoring corpora")
    return [
        row for row in measurements
        if not any(
            other["pack_bytes"] <= row["pack_bytes"] and other["mean_kl"] <= row["mean_kl"]
            and (other["pack_bytes"] < row["pack_bytes"] or other["mean_kl"] < row["mean_kl"])
            for other in measurements
        )
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--verify-source-hash", action="store_true",
                        help="stream all source shards and compare pack.py's aggregate provenance checksum")
    parser.add_argument("--measurements", type=Path,
                        help="JSON array of measured candidate bytes/KL with teacher/corpus identities")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.verify_source_hash and not args.source:
        parser.error("--verify-source-hash requires --source")
    pack = args.pack.expanduser()
    if pack.is_dir():
        pack /= "model.qwp"
    try:
        header, _ = read_header(pack)
        original_bytes = pack.stat().st_size
        candidates = [
            candidate_cost(header, original_bytes, (db, dg), (eb, eg))
            for db in (4, 6, 8) for dg in VALID_GROUPS
            for eb in VALID_BITS for eg in VALID_GROUPS
        ]
        measured = json.loads(args.measurements.read_text()) if args.measurements else []
        if not isinstance(measured, list):
            raise ValueError("measurements must be a JSON array")
        frontier = pareto_frontier(measured)
        report = {
            "version": 1,
            "status": "measured-frontier" if measured else "size-only",
            "source_pack": str(pack.resolve()),
            "source_pack_bytes": original_bytes,
            "source_provenance": header.get("provenance", {}),
            "estimate_caveat": "Preserves original padding/header budget; final alignment and metadata "
                               "may differ. Raw planes and external tables are unchanged. No KL inferred.",
            "candidates": sorted(candidates, key=lambda row: row["estimated_pack_bytes"]),
            "measured_frontier": frontier,
            "winner": None,
        }
        if args.source:
            checksum = header.get("provenance", {}).get("sha256_source") if args.verify_source_hash else None
            if args.verify_source_hash and checksum is None:
                raise ValueError("incumbent pack has no source checksum to verify")
            report["source_audit"] = audit_source(args.source.expanduser(), checksum)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        encoded = json.dumps(report, indent=2, allow_nan=False)
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
        print(f"{report['status']}: {len(candidates)} size estimates; "
              f"{len(frontier)} measured non-dominated candidates -> {args.output}")
        if report.get("source_audit", {}).get("status") == "source-checksum-mismatch":
            print("source checksum differs from incumbent provenance; see saved source_audit", file=sys.stderr)
            return 2
        return 1 if report.get("source_audit", {}).get("status") == "incomplete" else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"hq_frontier: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
